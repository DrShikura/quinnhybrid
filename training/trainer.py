"""
Training loop for QuINN Phase 1.

The training loop:
  1. For each batch: sample random prefix fractions (done in dataset)
  2. Forward pass: QuINN predicts complete waveform + length from prefix
  3. Losses: waveform L2 + length smooth-L1 (log scale)
  4. Metrics logged: loss components, length accuracy at ±15% threshold

Validation evaluates the Phase 1 success criterion:
  > QuINN's token count estimates at 50% prefix converge to within ±15% of
  > true length significantly more reliably than the baseline.
The baseline predicts the mean training sequence length for every file.
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
from data import PythonCodeDataset, collate_fn
from training.config import QuINNConfig


class QuINNTrainer:
    def __init__(self, config: QuINNConfig):
        self.cfg = config
        self.device = torch.device(config.resolve_device())
        print(f"Using device: {self.device}")

        self._set_seed(config.seed)
        self._build_data()
        self._build_model()
        self._build_optim()

        self.step = 0
        self.epoch = 0
        self.history = []

    # ── Setup ─────────────────────────────────────────────────────────────

    def _set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_data(self):
        cfg = self.cfg
        print("Loading corpus…")
        full_dataset = PythonCodeDataset(
            cfg.corpus_file,
            min_seq_len=cfg.min_seq_len,
            max_seq_len=cfg.max_seq_len_data,
            min_prefix_frac=cfg.min_prefix_frac,
            max_prefix_frac=cfg.max_prefix_frac,
        )
        n = len(full_dataset)
        print(f"  {n} files loaded")

        n_train = int(n * cfg.train_split)
        n_val   = int(n * cfg.val_split)
        n_test  = n - n_train - n_val

        self.train_set, self.val_set, self.test_set = random_split(
            full_dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(cfg.seed),
        )

        # Store mean training length for baseline comparison
        train_lens = [full_dataset.samples[i] for i in self.train_set.indices]
        self.mean_train_len = sum(len(s) for s in train_lens) / max(len(train_lens), 1)
        print(f"  Mean training sequence length: {self.mean_train_len:.1f} tokens")
        print(f"  Train/Val/Test: {n_train}/{n_val}/{n_test}")

        self.train_loader = DataLoader(
            self.train_set, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0, pin_memory=(self.device.type == "cuda"),
        )
        self.val_loader = DataLoader(
            self.val_set, batch_size=cfg.batch_size * 2, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )
        self.test_loader = DataLoader(
            self.test_set, batch_size=cfg.batch_size * 2, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
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

        self.loss_fn = QuINNLoss(alpha=self.cfg.loss_alpha)

    def _build_optim(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        # Cosine annealing with linear warmup
        total_steps = self.cfg.n_epochs * max(1, len(self.train_loader))
        warmup = self.cfg.warmup_steps

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ── Training step ─────────────────────────────────────────────────────

    def _step(self, batch: dict) -> dict:
        b = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

        pred_wf, true_wf, pred_log_len = self.model(
            prefix_tokens=b["prefix_tokens"],
            prefix_positions=b["prefix_positions"],
            prefix_mask=b["prefix_mask"],
            target_tokens=b["target_tokens"],
            target_positions=b["target_positions"],
        )

        losses = self.loss_fn(
            predicted_waveform=pred_wf,
            true_waveform=true_wf,
            pred_log_len=pred_log_len,
            true_len=b["seq_len"],
            target_mask=b["target_mask"],
        )
        return losses

    # ── Epoch loops ──────────────────────────────────────────────────────

    def train_epoch(self) -> dict:
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
                    f"lr={lr:.2e}"
                )

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, prefix_frac_filter: Optional[float] = None) -> dict:
        """
        Evaluate on loader. If prefix_frac_filter is set (e.g. 0.5), only
        uses samples with prefix fraction near that value (±0.1) to measure
        the Phase 1 success criterion at 50% prefix.
        """
        self.model.eval()
        totals = {}
        n_batches = 0
        n_within_15pct = 0
        n_baseline_within_15pct = 0
        n_total = 0

        for batch in loader:
            # Filter by prefix fraction if requested
            if prefix_frac_filter is not None:
                frac = batch["prefix_frac"]
                keep = (frac >= prefix_frac_filter - 0.1) & (frac <= prefix_frac_filter + 0.1)
                if keep.sum() == 0:
                    continue
                batch = {
                    k: v[keep] if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

            b = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

            pred_wf, true_wf, pred_log_len = self.model(
                prefix_tokens=b["prefix_tokens"],
                prefix_positions=b["prefix_positions"],
                prefix_mask=b["prefix_mask"],
                target_tokens=b["target_tokens"],
                target_positions=b["target_positions"],
            )

            losses = self.loss_fn(pred_wf, true_wf, pred_log_len, b["seq_len"], b["target_mask"])
            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + v.item()
            n_batches += 1

            # Length accuracy metrics
            true_len = b["seq_len"].float()
            pred_len = torch.exp(pred_log_len)
            baseline_len = torch.full_like(true_len, self.mean_train_len)

            within_15 = ((pred_len - true_len).abs() / true_len.clamp(min=1)) < 0.15
            baseline_within_15 = ((baseline_len - true_len).abs() / true_len.clamp(min=1)) < 0.15

            n_within_15pct += within_15.sum().item()
            n_baseline_within_15pct += baseline_within_15.sum().item()
            n_total += true_len.shape[0]

        metrics = {k: v / max(n_batches, 1) for k, v in totals.items()}
        if n_total > 0:
            metrics["length_acc_15pct"]          = n_within_15pct / n_total
            metrics["baseline_acc_15pct"]        = n_baseline_within_15pct / n_total
            metrics["beats_baseline"]            = (
                n_within_15pct / n_total > n_baseline_within_15pct / n_total
            )
        return metrics

    # ── Full training run ─────────────────────────────────────────────────

    def train(self):
        cfg = self.cfg
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"QuINN Phase 1 Training")
        print(f"{'='*60}")

        for epoch in range(1, cfg.n_epochs + 1):
            self.epoch = epoch
            t0 = time.time()
            print(f"\nEpoch {epoch}/{cfg.n_epochs}")

            train_metrics = self.train_epoch()
            val_metrics   = self.evaluate(self.val_loader)

            # Evaluate specifically at 50% prefix for the success criterion
            val_50_metrics = self.evaluate(self.val_loader, prefix_frac_filter=0.5)

            elapsed = time.time() - t0

            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val":   val_metrics,
                "val_50pct_prefix": val_50_metrics,
                "elapsed_s": elapsed,
            }
            self.history.append(record)

            acc   = val_metrics.get("length_acc_15pct", 0)
            base  = val_metrics.get("baseline_acc_15pct", 0)
            acc50 = val_50_metrics.get("length_acc_15pct", 0)

            print(
                f"  val_loss={val_metrics['loss']:.4f}  "
                f"len_acc={acc:.3f}  baseline={base:.3f}  "
                f"beats_baseline={'✓' if acc > base else '✗'}  "
                f"len_acc@50%={acc50:.3f}  "
                f"elapsed={elapsed:.1f}s"
            )

            # Save checkpoint
            if epoch % cfg.save_every_n_epochs == 0 or epoch == cfg.n_epochs:
                self.save_checkpoint(epoch)

        # Final test evaluation
        print("\n── Final test evaluation ─────────────────────────────────")
        test_metrics = self.evaluate(self.test_loader)
        for prefix_frac in [0.10, 0.30, 0.50, 0.70]:
            m = self.evaluate(self.test_loader, prefix_frac_filter=prefix_frac)
            acc   = m.get("length_acc_15pct", 0)
            base  = m.get("baseline_acc_15pct", 0)
            print(
                f"  prefix={prefix_frac:.0%}  "
                f"QuINN={acc:.3f}  baseline={base:.3f}  "
                f"beats={'✓' if acc > base else '✗'}"
            )

        self.save_history()
        return self.history

    # ── Persistence ──────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int):
        path = Path(self.cfg.checkpoint_dir) / f"quinn_epoch{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "step": self.step,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.cfg.__dict__,
                "mean_train_len": self.mean_train_len,
            },
            path,
        )
        print(f"  Saved checkpoint → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.step  = ckpt.get("step", 0)
        self.epoch = ckpt.get("epoch", 0)
        self.mean_train_len = ckpt.get("mean_train_len", self.mean_train_len)
        print(f"Loaded checkpoint from {path} (epoch {self.epoch})")

    def save_history(self):
        path = Path(self.cfg.checkpoint_dir) / "training_history.json"
        # Convert tensors to Python scalars for serialisation
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
