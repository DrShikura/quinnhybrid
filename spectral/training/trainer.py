"""
SpectralLM trainer.

Supports three training modes:
    waveform:  pre-train on waveform completion (learns frequency structure)
    lm:        train on next-token prediction only
    joint:     both losses — waveform as regularizer, LM as primary objective
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


def build_model(config: dict, vocab: list) -> SpectralLM:
    return SpectralLM(
        vocab       = vocab,
        embed_dim   = config.get('embed_dim',   64),
        n_layers    = config.get('n_layers',     4),
        n_heads     = config.get('n_heads',      8),
        max_seq_len = config.get('max_seq_len', 512),
        dropout     = config.get('dropout',    0.1),
    )


class SpectralTrainer:
    """
    Trainer for SpectralLM.

    Args:
        model:         SpectralLM instance
        train_loader:  DataLoader yielding (token_ids, position_ids) batches
        val_loader:    DataLoader for validation
        config:        training configuration dict
        checkpoint_dir: where to save checkpoints
    """

    def __init__(
        self,
        model:          SpectralLM,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        config:         dict,
        checkpoint_dir: str | Path,
        tokenizer,
    ):
        self.model    = model
        self.train_dl = train_loader
        self.val_dl   = val_loader
        self.config   = config
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer

        self.mode    = config.get('mode', 'joint')
        self.device  = torch.device('cpu')
        self.model.to(self.device)

        # Loss
        emb = model.embedding
        self.criterion = SpectralLoss(
            embed_dim       = model.embed_dim,
            structural_end  = emb.structural_end,
            expr_end        = emb.expr_end,
            lm_weight       = config.get('lm_weight',       1.0),
            waveform_weight = config.get('waveform_weight', 0.5),
            band_weights    = tuple(config.get('band_weights', [1.5, 1.0, 0.5])),
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

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr          = self.lr,
            total_steps     = total_steps,
            pct_start       = warmup_steps / total_steps,
            anneal_strategy = 'cos',
        )

        self.history = []

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict:
        self.model.train(train)
        total_loss = 0.0
        total_lm   = 0.0
        total_wave = 0.0
        n_steps    = 0

        with torch.set_grad_enabled(train):
            for batch in loader:
                token_ids, position_ids = batch
                token_ids    = token_ids.to(self.device)
                position_ids = position_ids.to(self.device)

                # Forward
                out = self.model(token_ids, position_ids, mode=self.mode)

                # For waveform target: compute embedding of full sequence
                # (shift by 1 to get next-token waveform)
                if self.mode in ('waveform', 'joint'):
                    with torch.no_grad():
                        full_wave = self.model.embedding(token_ids, position_ids)
                else:
                    full_wave = None

                # Loss
                losses = self.criterion(
                    model_out   = out,
                    target_ids  = token_ids,
                    target_wave = full_wave,
                    pad_id      = self.tokenizer.pad_id,
                    mode        = self.mode,
                )

                if train:
                    self.optimizer.zero_grad()
                    losses['total'].backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()

                total_loss += losses['total'].item()
                if 'lm'          in losses: total_lm   += losses['lm'].item()
                if 'wave_total'  in losses: total_wave += losses['wave_total'].item()
                n_steps += 1

        return {
            'loss':       total_loss / max(n_steps, 1),
            'lm_loss':    total_lm   / max(n_steps, 1),
            'wave_loss':  total_wave / max(n_steps, 1),
        }

    def train(self):
        counts = self.model.param_count()
        print(f"\nSpectralLM parameters: {counts['total']:,}")
        print(f"  Embedding:   {counts['embedding']:,}")
        print(f"  Transformer: {counts['transformer']:,}")
        print(f"  Heads:       {counts['heads']:,}")
        print(f"\nMode: {self.mode}   Epochs: {self.n_epochs}")
        print(f"Structural band: dims 0-{self.model.embedding.structural_end}")
        print(f"Expression band: dims {self.model.embedding.structural_end}-{self.model.embedding.expr_end}")
        print(f"Semantic band:   dims {self.model.embedding.expr_end}-{self.model.embed_dim}")
        print("=" * 60)

        start_epoch = getattr(self, '_start_epoch', 1)
        # Restore best_val from history if resuming
        if self.history:
            best_val = min(r['val']['loss'] for r in self.history)
        else:
            best_val = float('inf')
        t0 = time.time()

        for epoch in range(start_epoch, self.n_epochs + 1):
            train_metrics = self._run_epoch(self.train_dl, train=True)
            val_metrics   = self._run_epoch(self.val_dl,   train=False)

            elapsed = time.time() - t0
            lr_now  = self.scheduler.get_last_lr()[0]

            print(f"\nEpoch {epoch}/{self.n_epochs}  "
                  f"lr={lr_now:.2e}  elapsed={elapsed:.0f}s")
            print(f"  Train — loss={train_metrics['loss']:.4f}  "
                  f"lm={train_metrics['lm_loss']:.4f}  "
                  f"wave={train_metrics['wave_loss']:.4f}")
            print(f"  Val   — loss={val_metrics['loss']:.4f}  "
                  f"lm={val_metrics['lm_loss']:.4f}  "
                  f"wave={val_metrics['wave_loss']:.4f}")

            record = {
                'epoch': epoch,
                'train': train_metrics,
                'val':   val_metrics,
                'lr':    lr_now,
            }
            self.history.append(record)

            # Checkpoint best model
            if val_metrics['loss'] < best_val:
                best_val = val_metrics['loss']
                self._save(epoch, val_metrics['loss'], 'best')

            # Periodic checkpoint
            if epoch % 10 == 0:
                self._save(epoch, val_metrics['loss'], f'epoch{epoch:03d}')

        # Final checkpoint + history
        self._save(self.n_epochs, self.history[-1]['val']['loss'], 'final')
        with open(self.ckpt_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining complete. Best val loss: {best_val:.4f}")
        return self.history

    def _save(self, epoch: int, val_loss: float, tag: str):
        path = self.ckpt_dir / f'spectral_{tag}.pt'
        torch.save({
            'epoch':        epoch,
            'val_loss':     val_loss,
            'model_state':  self.model.state_dict(),
            'opt_state':    self.optimizer.state_dict(),
            'sched_state':  self.scheduler.state_dict(),
            'history':      self.history,
            'config':       self.config,
        }, path)

    def resume(self, checkpoint_path: str):
        """Load model + optimizer + scheduler state to continue training."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['opt_state'])
        self.scheduler.load_state_dict(ckpt['sched_state'])
        self.history = ckpt.get('history', [])
        start_epoch  = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")
        return start_epoch
