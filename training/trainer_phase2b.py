"""
Phase 2B training: Hybrid QuINN/Transformer.

QuINN predicts structural waveform. Transformer uses it as guidance to predict tokens.

Training objective:
  - Language modeling: predict next token given prefix
  - Waveform reconstruction (QuINN): auxiliary task
  - Optional: length prediction

Architecture:
  - QuINN encoder: prefix → waveform summary
  - Transformer decoder: (prefix_tokens, quinn_waveform) → token logits
  - Trained end-to-end
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

from quinn import QuINN
from quinn.transformer import StructuralTransformerDecoder
from data import collate_fn
from data.dataset import PythonCodeDataset
from training.config import QuINNConfig


class QuINNTransformerDataset(PythonCodeDataset):
    """
    Dataset for Phase 2B: predict tokens given prefix + Quinn waveform guidance.

    Each sample: prefix → target sequence (full sequence)
    Task: predict target tokens autoregressively
    Guidance: Quinn waveform from prefix
    """

    def __getitem__(self, idx: int) -> dict:
        ids = self.samples[idx]
        n = len(ids)

        # Sample prefix fraction
        frac = self.rng.uniform(self.min_prefix_frac, self.max_prefix_frac)
        prefix_len = max(2, int(round(n * frac)))
        prefix_len = min(prefix_len, n - 1)

        prefix_ids = ids[:prefix_len]
        all_ids = ids

        return {
            "prefix_ids": torch.tensor(prefix_ids, dtype=torch.long),
            "prefix_positions": torch.arange(prefix_len, dtype=torch.long),
            "target_ids": torch.tensor(all_ids, dtype=torch.long),
            "target_positions": torch.arange(n, dtype=torch.long),
            "seq_len": torch.tensor(n, dtype=torch.long),
            "prefix_frac": torch.tensor(frac, dtype=torch.float),
        }


def collate_fn_phase2b(batch: list) -> dict:
    """Collate for phase 2b: prefix and target sequences."""
    from data.tokenizer import PAD_ID

    max_prefix = max(b["prefix_ids"].shape[0] for b in batch)
    max_target = max(b["target_ids"].shape[0] for b in batch)

    prefix_ids = torch.full((len(batch), max_prefix), PAD_ID, dtype=torch.long)
    prefix_pos = torch.zeros(len(batch), max_prefix, dtype=torch.long)
    prefix_mask = torch.zeros(len(batch), max_prefix, dtype=torch.bool)

    target_ids = torch.full((len(batch), max_target), PAD_ID, dtype=torch.long)
    target_pos = torch.zeros(len(batch), max_target, dtype=torch.long)
    target_mask = torch.zeros(len(batch), max_target, dtype=torch.bool)

    seq_lens = torch.stack([b["seq_len"] for b in batch])
    prefix_fracs = torch.stack([b["prefix_frac"] for b in batch])

    for i, b in enumerate(batch):
        p = b["prefix_ids"].shape[0]
        t = b["target_ids"].shape[0]

        prefix_ids[i, :p] = b["prefix_ids"]
        prefix_pos[i, :p] = b["prefix_positions"]
        prefix_mask[i, :p] = True

        target_ids[i, :t] = b["target_ids"]
        target_pos[i, :t] = b["target_positions"]
        target_mask[i, :t] = True

    return {
        "prefix_tokens": prefix_ids,
        "prefix_positions": prefix_pos,
        "prefix_mask": prefix_mask,
        "target_tokens": target_ids,
        "target_positions": target_pos,
        "target_mask": target_mask,
        "seq_len": seq_lens,
        "prefix_frac": prefix_fracs,
    }


class QuINNTransformerTrainer:
    """Phase 2B: train Transformer guided by Quinn waveform."""

    def __init__(self, config: QuINNConfig, quinn_checkpoint: Optional[str] = None):
        self.cfg = config
        self.device = torch.device(config.resolve_device())
        print(f"Using device: {self.device}")

        self._set_seed(config.seed)
        self._build_data()
        self._build_model(quinn_checkpoint)
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
        print("Loading corpus (Phase 2B)…")
        full_dataset = QuINNTransformerDataset(
            cfg.corpus_file,
            min_seq_len=cfg.min_seq_len,
            max_seq_len=cfg.max_seq_len_data,
            min_prefix_frac=cfg.min_prefix_frac,
            max_prefix_frac=cfg.max_prefix_frac,
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
        print(f"  Train/Val/Test: {n_train}/{n_val}/{n_test}")

        self.train_loader = DataLoader(
            self.train_set, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=collate_fn_phase2b, num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )
        self.val_loader = DataLoader(
            self.val_set, batch_size=cfg.batch_size * 2, shuffle=False,
            collate_fn=collate_fn_phase2b, num_workers=0,
        )

    def _build_model(self, quinn_checkpoint: Optional[str]):
        from data.tokenizer import PythonStructuralTokenizer
        tok = PythonStructuralTokenizer()
        vocab_size = tok.vocab_size

        # QuINN (frozen or trainable)
        print("Loading Quinn…")
        self.quinn = QuINN(
            vocab_size=vocab_size,
            embed_dim=self.cfg.embed_dim,
            manifold_dim=self.cfg.manifold_dim,
            n_encoder_layers=self.cfg.n_encoder_layers,
            n_decoder_layers=self.cfg.n_decoder_layers,
            max_seq_len=self.cfg.max_seq_len,
        ).to(self.device)

        if quinn_checkpoint:
            print(f"  Loading Quinn from {quinn_checkpoint}")
            ckpt = torch.load(quinn_checkpoint, map_location=self.device)
            self.quinn.load_state_dict(ckpt["model_state"])
        else:
            print(f"  Quinn initialized randomly (no checkpoint)")

        # Freeze Quinn for now (can unfreeze later for fine-tuning)
        for p in self.quinn.parameters():
            p.requires_grad = False
        self.quinn.eval()

        # Transformer decoder
        print("Creating Transformer…")
        self.transformer = StructuralTransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=self.cfg.embed_dim,
            hidden_dim=self.cfg.embed_dim * 4,
            n_heads=8,
            n_layers=4,
            max_seq_len=self.cfg.max_seq_len,
        ).to(self.device)

        print(f"Quinn parameters: {self.quinn.count_parameters():,}")
        print(f"Transformer parameters: {sum(p.numel() for p in self.transformer.parameters()):,}")

        # Loss functions
        self.lm_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding

    def _build_optim(self):
        # Only optimize Transformer (Quinn frozen)
        self.optimizer = torch.optim.AdamW(
            self.transformer.parameters(),
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
        """Single training step."""
        b = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

        # Get Quinn waveform from prefix (no grad)
        with torch.no_grad():
            _, _, _ = self.quinn(
                b["prefix_tokens"],
                b["prefix_positions"],
                b["prefix_mask"],
            )
            # Get the waveform explicitly
            waveform = self.quinn.decoder(
                self.quinn._encode_and_aggregate(
                    b["prefix_tokens"],
                    b["prefix_positions"],
                    b["prefix_mask"],
                ),
                b["target_positions"],  # use target positions for guidance length
            )  # (B, T, embed_dim) complex

        # Transformer: predict tokens using Quinn waveform as guidance
        logits = self.transformer(
            token_ids=b["target_tokens"],
            positions=b["target_positions"],
            quinn_waveform=waveform,
            mask=b["target_mask"],
        )  # (B, T, vocab_size)

        # Language modeling loss
        # Shift for autoregressive: predict token i+1 given tokens 0..i
        logits_shifted = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
        targets_shifted = b["target_tokens"][:, 1:].contiguous().view(-1)

        loss_lm = self.lm_loss_fn(logits_shifted, targets_shifted)

        return {
            "loss": loss_lm,
            "lm_loss": loss_lm,
        }

    def train_epoch(self) -> dict:
        """Train one epoch."""
        self.transformer.train()
        totals = {}
        n_batches = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()
            losses = self._step(batch)
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(self.transformer.parameters(), self.cfg.grad_clip)
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
                    f"lm_loss={totals['lm_loss']/n_batches:.4f}  "
                    f"lr={lr:.2e}"
                )

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        """Evaluate on a loader."""
        self.transformer.eval()
        totals = {}
        n_batches = 0

        for batch in loader:
            b = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

            # Quinn waveform
            waveform = self.quinn.decoder(
                self.quinn._encode_and_aggregate(
                    b["prefix_tokens"],
                    b["prefix_positions"],
                    b["prefix_mask"],
                ),
                b["target_positions"],
            )

            # Transformer prediction
            logits = self.transformer(
                token_ids=b["target_tokens"],
                positions=b["target_positions"],
                quinn_waveform=waveform,
                mask=b["target_mask"],
            )

            # LM loss
            logits_shifted = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
            targets_shifted = b["target_tokens"][:, 1:].contiguous().view(-1)
            loss_lm = self.lm_loss_fn(logits_shifted, targets_shifted)

            totals["lm_loss"] = totals.get("lm_loss", 0.0) + loss_lm.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def train(self):
        """Full training run."""
        cfg = self.cfg
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"QuINN/Transformer Phase 2B Training")
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
                f"  val_lm_loss={val_metrics['lm_loss']:.4f}  "
                f"elapsed={elapsed:.1f}s"
            )

        self.save_history()
        print(f"\n✓ Phase 2B training complete.")
        return self.history

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
