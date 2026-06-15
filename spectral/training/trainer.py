"""
SpectralLM trainer.

Supports three training modes:
    waveform:  pre-train on waveform completion (learns frequency structure)
    lm:        train on next-token prediction only
    joint:     both losses — waveform as regularizer, LM as primary objective

Loss schedule:
    waveform_weight: cosine-annealed from base → 0 over training epochs
    gradient_weight: CONSTANT throughout (frozen entropy target never changes)
"""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..model.spectral_lm import SpectralLM
from ..model.loss import SpectralLoss


def build_model(config: dict, vocab: list,
                laplacian_eigvecs=None) -> SpectralLM:
    return SpectralLM(
        vocab              = vocab,
        embed_dim          = config.get('embed_dim',     64),
        n_loops            = config.get('n_loops',  config.get('n_layers', 3)),
        n_heads            = config.get('n_heads',        8),
        max_seq_len        = config.get('max_seq_len',  512),
        dropout            = config.get('dropout',      0.1),
        band_init          = config.get('band_init',    True),
        acoustic_init      = config.get('acoustic_init', True),
        laplacian_eigvecs  = laplacian_eigvecs,
        mem_dim            = config.get('mem_dim',       32),
    )


class SpectralTrainer:
    """
    Trainer for SpectralLM.

    Args:
        model:          SpectralLM instance
        train_loader:   DataLoader yielding (token_ids, position_ids) batches
        val_loader:     DataLoader for validation
        config:         training configuration dict
        checkpoint_dir: where to save checkpoints
        tokenizer:      for pad_id
        frozen_entropy: (vocab_size,) H_norm from SpectralDataset — passed to
                        WaveformGradientConsistencyLoss each step; if None, that
                        loss is disabled
    """

    def __init__(
        self,
        model:          SpectralLM,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        config:         dict,
        checkpoint_dir: str | Path,
        tokenizer,
        frozen_entropy: 'torch.Tensor | None' = None,
        device: 'torch.device | None' = None,
    ):
        self.model    = model
        self.train_dl = train_loader
        self.val_dl   = val_loader
        self.config   = config
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer

        self.mode = config.get('mode', 'joint')
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)

        # Frozen entropy tensor on same device
        self.frozen_entropy = (
            frozen_entropy.to(self.device) if frozen_entropy is not None else None
        )

        # Loss — SpectralLoss band params only meaningful for SpectralLM
        emb = getattr(model, 'embedding', None)
        embed_dim      = getattr(model, 'embed_dim', getattr(model, 'd_model', 32))
        structural_end = getattr(emb, 'structural_end', embed_dim // 4)
        expr_end       = getattr(emb, 'expr_end',       embed_dim // 2)
        self.criterion = SpectralLoss(
            embed_dim         = embed_dim,
            structural_end    = structural_end,
            expr_end          = expr_end,
            lm_weight         = config.get('lm_weight',        1.0),
            waveform_weight   = config.get('waveform_weight',  0.5),
            band_weights      = tuple(config.get('band_weights', [1.5, 1.0, 0.5])),
            gradient_weight   = config.get('gradient_weight',  0.1),
            entropy_threshold = config.get('entropy_threshold', 0.4),
            memory_weight     = config.get('memory_weight',    0.1),
        )

        # Optimizer with warmup + cosine decay
        self.n_epochs  = config.get('n_epochs', 30)
        self.lr        = config.get('lr', 3e-4)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr,
            weight_decay=config.get('weight_decay', 0.01),
        )

        steps_per_epoch = len(train_loader)
        total_steps     = self.n_epochs * steps_per_epoch
        warmup_steps    = config.get('warmup_steps', min(500, total_steps // 10))

        if total_steps >= 20:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr          = self.lr,
                total_steps     = total_steps,
                pct_start       = max(warmup_steps / total_steps, 1 / total_steps),
                anneal_strategy = 'cos',
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(total_steps, 1), eta_min=self.lr * 0.01
            )

        self.history = []

    def _wave_weight(self, epoch: int) -> float:
        """Cosine-anneal waveform loss weight: full at epoch 1, zero at end."""
        base_w   = self.config.get('waveform_weight', 0.5)
        schedule = self.config.get('wave_schedule', 'cosine')
        if schedule == 'none':
            return base_w
        progress = (epoch - 1) / max(self.n_epochs - 1, 1)
        if schedule == 'linear':
            return base_w * max(0.0, 1.0 - progress)
        return base_w * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict:
        self.model.train(train)
        total_loss = total_lm = total_wave = total_grad = total_mem = 0.0
        n_steps    = 0

        with torch.set_grad_enabled(train):
            for batch in loader:
                token_ids, position_ids = batch
                token_ids    = token_ids.to(self.device)
                position_ids = position_ids.to(self.device)

                out = self.model(token_ids, position_ids, mode=self.mode)

                # Amplitude target — only available for SpectralLM
                if self.mode in ('waveform', 'joint') and hasattr(self.model, 'embedding') and hasattr(self.model.embedding, 'amplitude'):
                    with torch.no_grad():
                        full_wave = self.model.embedding.amplitude(token_ids)
                else:
                    full_wave = None

                losses = self.criterion(
                    model_out      = out,
                    target_ids     = token_ids,
                    target_wave    = full_wave,
                    frozen_entropy = self.frozen_entropy,
                    pad_id         = self.tokenizer.pad_id,
                    mode           = self.mode,
                )

                if train:
                    self.optimizer.zero_grad()
                    losses['total'].backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()

                total_loss += losses['total'].item()
                if 'lm'         in losses: total_lm   += losses['lm'].item()
                if 'wave_total' in losses: total_wave += losses['wave_total'].item()
                if 'grad'       in losses: total_grad += losses['grad'].item()
                if 'mem'        in losses: total_mem  += losses['mem'].item()
                n_steps += 1

        return {
            'loss':      total_loss / max(n_steps, 1),
            'lm_loss':   total_lm   / max(n_steps, 1),
            'wave_loss': total_wave / max(n_steps, 1),
            'grad_loss': total_grad / max(n_steps, 1),
            'mem_loss':  total_mem  / max(n_steps, 1),
        }

    def train(self):
        counts     = self.model.param_count()
        model_name = type(self.model).__name__
        print(f"\n{model_name} parameters: {counts['total']:,}")
        print(f"  Embedding:   {counts['embedding']:,}")
        n_loops = getattr(self.model, 'n_loops', 1)
        if counts.get('memory', 0):
            print(f"  Transformer: {counts['transformer']:,}  (1 shared layer × {n_loops} loops)")
            print(f"  Memory:      {counts['memory']:,}")
            print(f"  Heads:       {counts['heads']:,}  (LM head partially tied to amplitude)")
        else:
            n_layers = n_loops
            print(f"  Transformer: {counts['transformer']:,}  ({n_layers} layer{'s' if n_layers > 1 else ''})")
        print(f"\nMode: {self.mode}   Epochs: {self.n_epochs}   Loops/step: {n_loops}")
        if hasattr(self.model, 'embedding') and hasattr(self.model.embedding, 'structural_end'):
            emb = self.model.embedding
            print(f"Structural band: dims 0-{emb.structural_end}")
            print(f"Expression band: dims {emb.structural_end}-{emb.expr_end}")
            print(f"Semantic band:   dims {emb.expr_end}-{self.model.embed_dim}")
            print(f"Gradient consistency loss weight: {self.criterion.gradient_weight:.3f} (constant)")
            print(f"Memory loss weight:               {self.criterion.memory_weight:.3f} (constant)")
        print("=" * 60)

        start_epoch = getattr(self, '_start_epoch', 1)
        best_val    = min((r['val']['loss'] for r in self.history), default=float('inf'))
        t0 = time.time()

        for epoch in range(start_epoch, self.n_epochs + 1):
            # Anneal waveform weight only
            self.criterion.waveform_weight = self._wave_weight(epoch)

            train_metrics = self._run_epoch(self.train_dl, train=True)
            val_metrics   = self._run_epoch(self.val_dl,   train=False)

            elapsed = time.time() - t0
            lr_now  = self.scheduler.get_last_lr()[0]
            wave_w  = self.criterion.waveform_weight

            print(f"\nEpoch {epoch}/{self.n_epochs}  "
                  f"lr={lr_now:.2e}  wave_w={wave_w:.3f}  elapsed={elapsed:.0f}s")
            print(f"  Train — loss={train_metrics['loss']:.4f}  "
                  f"lm={train_metrics['lm_loss']:.4f}  "
                  f"wave={train_metrics['wave_loss']:.4f}  "
                  f"grad={train_metrics['grad_loss']:.4f}  "
                  f"mem={train_metrics['mem_loss']:.4f}")
            print(f"  Val   — loss={val_metrics['loss']:.4f}  "
                  f"lm={val_metrics['lm_loss']:.4f}  "
                  f"wave={val_metrics['wave_loss']:.4f}  "
                  f"grad={val_metrics['grad_loss']:.4f}  "
                  f"mem={val_metrics['mem_loss']:.4f}")

            record = {'epoch': epoch, 'train': train_metrics,
                      'val': val_metrics, 'lr': lr_now}
            self.history.append(record)

            if val_metrics['loss'] < best_val:
                best_val = val_metrics['loss']
                self._save(epoch, val_metrics['loss'], 'best')

            if epoch % 10 == 0:
                self._save(epoch, val_metrics['loss'], f'epoch{epoch:03d}')

        self._save(self.n_epochs, self.history[-1]['val']['loss'], 'final')
        with open(self.ckpt_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining complete. Best val loss: {best_val:.4f}")
        return self.history

    def _save(self, epoch: int, val_loss: float, tag: str):
        path = self.ckpt_dir / f'spectral_{tag}.pt'
        torch.save({
            'epoch':       epoch,
            'val_loss':    val_loss,
            'model_state': self.model.state_dict(),
            'opt_state':   self.optimizer.state_dict(),
            'sched_state': self.scheduler.state_dict(),
            'history':     self.history,
            'config':      self.config,
        }, path)

    def resume(self, checkpoint_path: str) -> int:
        """Load model + optimizer state to continue training.

        The LR scheduler is rebuilt for the remaining epochs rather than
        restored from the checkpoint — restoring the old scheduler state would
        overwrite total_steps with the original run's value and cause a
        ValueError on the first step when training beyond that count.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)
        missing, _ = self.model.load_state_dict(ckpt['model_state'], strict=False)
        if missing:
            print(f"  (initializing {len(missing)} new params: "
                  f"{missing[:3]}{'...' if len(missing) > 3 else ''})")
        self.optimizer.load_state_dict(ckpt['opt_state'])

        # Rebuild scheduler for remaining epochs at a reduced peak LR
        start_epoch      = ckpt['epoch'] + 1
        remaining_epochs = self.n_epochs - ckpt['epoch']
        remaining_steps  = max(remaining_epochs * len(self.train_dl), 1)
        warmup_steps     = min(100, remaining_steps // 10)
        if remaining_steps >= 20:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr          = self.lr * 0.3,   # lower peak for continued training
                total_steps     = remaining_steps,
                pct_start       = max(warmup_steps / remaining_steps,
                                      1 / remaining_steps),
                anneal_strategy = 'cos',
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=remaining_steps, eta_min=self.lr * 0.01
            )

        self.history = ckpt.get('history', [])
        print(f"Resumed from epoch {ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")
        return start_epoch
