"""
Phase 2A training: Contrastive learning on waveforms.

Multi-task loss:
  1. Waveform completion (positive target)
  2. Length prediction
  3. Contrastive: pull positive waveform close, push negative far
"""

import os
import math
import time
import json
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from quinn import QuINN, QuINNLoss
from quinn.contrastive_loss import WaveformContrastiveLoss
from data import collate_fn
from data.dataset_contrastive import PythonCodeDatasetContrastive, collate_fn_contrastive
from training.config import QuINNConfig


class QuINNTrainerPhase2A:
    """Phase 2A: train QuINN with contrastive loss on waveforms."""

    def __init__(self, config: QuINNConfig, contrastive_weight: float = 1.0):
        self.cfg = config
        self.device = torch.device(config.resolve_device())
        self.contrastive_weight = contrastive_weight
        print(f"Using device: {self.device}")

        self._set_seed(config.seed)
        self._build_data()
        self._build_model()
        self._build_optim()

        self.step = 0
        self.epoch = 0
        self.history = []

    def _set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_data(self):
        cfg = self.cfg
        print("Loading corpus (contrastive)…")
        full_dataset = PythonCodeDatasetContrastive(
            cfg.corpus_file,
            min_seq_len=cfg.min_seq_len,
            max_seq_len=cfg.max_seq_len_data,
            min_prefix_frac=cfg.min_prefix_frac,
            max_prefix_frac=cfg.max_prefix_frac,
            neg_num_swaps=2,  # swap 1-2 tokens
        )
        n = len(full_dataset)
        print(f"  {n} files loaded")

        n_train = int(n * cfg.train_split)
        n_val = int(n * cfg.val_split)
        n_test = n - n_train - n_val

        self.train_set, self.val_set, self.test_set = random_split(
            full_dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(cfg.seed),
        )

        # Store mean training length for baseline
        train_lens = [full_dataset.samples[i] for i in self.train_set.indices]
        self.mean_train_len = sum(len(s) for s in train_lens) / max(len(train_lens), 1)
        print(f"  Mean training sequence length: {self.mean_train_len:.1f} tokens")
        print(f"  Train/Val/Test: {n_train}/{n_val}/{n_test}")

        self.train_loader = DataLoader(
            self.train_set, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=collate_fn_contrastive, num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )
        self.val_loader = DataLoader(
            self.val_set, batch_size=cfg.batch_size * 2, shuffle=False,
            collate_fn=collate_fn_contrastive, num_workers=0,
        )

    def _build_model(self):
        from data.tokenizer import PythonStructuralTokenizer
        tok = PythonStructuralTokenizer()

        self.model = QuINN(
            vocab_size=tok.vocab_size,
            embed_dim=self.cfg.embed_dim,
            manifold_dim=self.cfg.manifold_dim,
            n_encoder_layers=self.cfg.n_encoder_layers,
            n_decoder_layers=self.cfg.n_decoder_layers,
            max_seq_len=self.cfg.max_seq_len,
        ).to(self.device)

        n_params = self.model.count_parameters()
        print(f"QuINN parameters: {n_params:,}")

        # Phase 1 losses
        self.waveform_loss_fn = QuINNLoss(alpha=self.cfg.loss_alpha)

        # Phase 2A loss
        self.contrastive_loss_fn = WaveformContrastiveLoss(temperature=0.07)

    def _build_optim(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        total_steps = self.cfg.n_epochs * max(1, len(self.train_loader))
        warmup = self.cfg.warmup_steps

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _step(self, batch: dict) -> dict:
        """Single training step on a contrastive batch."""
        b = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

        # Forward pass on positive target
        pred_wf_pos, true_wf_pos, pred_log_len = self.model(
            prefix_tokens=b["prefix_tokens"],
            prefix_positions=b["prefix_positions"],
            prefix_mask=b["prefix_mask"],
            target_tokens=b["target_tokens_positive"],
            target_positions=b["target_positions_positive"],
            prefix_frac=b["prefix_frac"],
        )

        # Forward pass on negative target (encoder + decoder, but no loss yet)
        pred_wf_neg, true_wf_neg, _ = self.model(
            prefix_tokens=b["prefix_tokens"],
            prefix_positions=b["prefix_positions"],
            prefix_mask=b["prefix_mask"],
            target_tokens=b["target_tokens_negative"],
            target_positions=b["target_positions_negative"],
            prefix_frac=b["prefix_frac"],
        )

        # Encode the prefix to get summary (anchor for contrastive loss)
        prefix_summary = self.model._encode_and_aggregate(
            b["prefix_tokens"],
            b["prefix_positions"],
            b["prefix_mask"],
        )

        # Encode positive and negative full sequences to get their summaries
        positive_summary = self.model._encode_and_aggregate(
            b["target_tokens_positive"],
            b["target_positions_positive"],
            b["target_mask_positive"],
        )
        negative_summary = self.model._encode_and_aggregate(
            b["target_tokens_negative"],
            b["target_positions_negative"],
            b["target_mask_negative"],
        )

        # Phase 1 losses (waveform + length)
        losses_phase1 = self.waveform_loss_fn(
            predicted_waveform=pred_wf_pos,
            true_waveform=true_wf_pos,
            pred_log_len=pred_log_len,
            true_len=b["seq_len"],
            target_mask=b["target_mask_positive"],
        )

        # Phase 2A loss (contrastive)
        loss_contrastive = self.contrastive_loss_fn(
            prefix_summary=prefix_summary,
            positive_summary=positive_summary,
            negative_summary=negative_summary,
        )

        # Combined loss
        loss_total = losses_phase1["loss"] + self.contrastive_weight * loss_contrastive

        return {
            "loss": loss_total,
            "waveform_loss": losses_phase1["waveform_loss"],
            "length_loss": losses_phase1["length_loss"],
            "contrastive_loss": loss_contrastive,
        }

    def train_epoch(self) -> dict:
        """Train one epoch."""
        self.model.train()
        totals = {}
        n_batches = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()
            losses = self._step(batch)
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + v.item()
            n_batches += 1
            self.step += 1

            if self.step % self.cfg.log_every_n_steps == 0:
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"  step={self.step:5d}  "
                    f"loss={totals['loss']/n_batches:.4f}  "
                    f"wf={totals['waveform_loss']/n_batches:.4f}  "
                    f"len={totals['length_loss']/n_batches:.4f}  "
                    f"contra={totals['contrastive_loss']/n_batches:.4f}  "
                    f"lr={lr:.2e}"
                )

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        """Evaluate on a loader."""
        self.model.eval()
        totals = {}
        n_batches = 0

        for batch in loader:
            b = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

            # Forward passes
            pred_wf_pos, true_wf_pos, pred_log_len = self.model(
                prefix_tokens=b["prefix_tokens"],
                prefix_positions=b["prefix_positions"],
                prefix_mask=b["prefix_mask"],
                target_tokens=b["target_tokens_positive"],
                target_positions=b["target_positions_positive"],
                prefix_frac=b["prefix_frac"],
            )

            pred_wf_neg, true_wf_neg, _ = self.model(
                prefix_tokens=b["prefix_tokens"],
                prefix_positions=b["prefix_positions"],
                prefix_mask=b["prefix_mask"],
                target_tokens=b["target_tokens_negative"],
                target_positions=b["target_positions_negative"],
                prefix_frac=b["prefix_frac"],
            )

            prefix_summary = self.model._encode_and_aggregate(
                b["prefix_tokens"],
                b["prefix_positions"],
                b["prefix_mask"],
            )
            positive_summary = self.model._encode_and_aggregate(
                b["target_tokens_positive"],
                b["target_positions_positive"],
                b["target_mask_positive"],
            )
            negative_summary = self.model._encode_and_aggregate(
                b["target_tokens_negative"],
                b["target_positions_negative"],
                b["target_mask_negative"],
            )

            # Losses
            losses_phase1 = self.waveform_loss_fn(
                predicted_waveform=pred_wf_pos,
                true_waveform=true_wf_pos,
                pred_log_len=pred_log_len,
                true_len=b["seq_len"],
                target_mask=b["target_mask_positive"],
            )
            loss_contrastive = self.contrastive_loss_fn(
                prefix_summary=prefix_summary,
                positive_summary=positive_summary,
                negative_summary=negative_summary,
            )
            loss_total = losses_phase1["loss"] + self.contrastive_weight * loss_contrastive

            for k, v in {"loss": loss_total, "waveform_loss": losses_phase1["waveform_loss"],
                        "length_loss": losses_phase1["length_loss"],
                        "contrastive_loss": loss_contrastive}.items():
                totals[k] = totals.get(k, 0.0) + v.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def train(self):
        """Full training run."""
        cfg = self.cfg
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"QuINN Phase 2A Training (Contrastive)")
        print(f"{'='*60}")

        for epoch in range(1, cfg.n_epochs + 1):
            self.epoch = epoch
            t0 = time.time()
            print(f"\nEpoch {epoch}/{cfg.n_epochs}")

            train_metrics = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)

            elapsed = time.time() - t0

            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "elapsed_s": elapsed,
            }
            self.history.append(record)

            print(
                f"  val_loss={val_metrics['loss']:.4f}  "
                f"wf={val_metrics['waveform_loss']:.4f}  "
                f"len={val_metrics['length_loss']:.4f}  "
                f"contra={val_metrics['contrastive_loss']:.4f}  "
                f"elapsed={elapsed:.1f}s"
            )

        self.save_history()
        print(f"\n✓ Phase 2A training complete.")
        return self.history

    def save_checkpoint(self, epoch: int):
        path = Path(self.cfg.checkpoint_dir) / f"quinn_epoch{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.cfg.__dict__,
            "mean_train_len": self.mean_train_len,
        }, path)
        print(f"  Saved checkpoint → {path}")

    def save_history(self):
        path = Path(self.cfg.checkpoint_dir) / "training_history.json"
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(x) for x in obj]
            if isinstance(obj, torch.Tensor):
                return obj.item()
            if isinstance(obj, bool):
                return obj
            if isinstance(obj, (float, int)):
                return obj
            return str(obj)

        with open(path, "w") as f:
            json.dump(_clean(self.history), f, indent=2)
        print(f"Training history → {path}")
