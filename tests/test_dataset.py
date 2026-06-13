"""Tests for data.dataset module."""

import sys
import os
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from data.dataset import PythonCodeDataset, collate_fn
from data.tokenizer import PythonStructuralTokenizer, PAD_ID, BOS_ID, EOS_ID


SAMPLE_PY = """
def add(a, b):
    return a + b

class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

    def reset(self):
        self.count = 0
"""


@pytest.fixture
def tmp_py_files(tmp_path):
    """Create a small corpus of Python files in a temp directory."""
    files = []
    for i in range(10):
        p = tmp_path / f"file_{i}.py"
        # Make files of varying lengths
        content = SAMPLE_PY * (i + 1)
        p.write_text(content)
        files.append(p)
    return files


@pytest.fixture
def small_dataset(tmp_py_files):
    return PythonCodeDataset(
        tmp_py_files,
        min_seq_len=5,
        max_seq_len=512,
        min_prefix_frac=0.2,
        max_prefix_frac=0.8,
        seed=42,
    )


class TestPythonCodeDataset:
    def test_loads_files(self, small_dataset):
        assert len(small_dataset) > 0

    def test_getitem_keys(self, small_dataset):
        sample = small_dataset[0]
        required_keys = {"prefix_ids", "prefix_positions", "all_ids", "all_positions", "seq_len"}
        assert required_keys.issubset(sample.keys())

    def test_prefix_shorter_than_all(self, small_dataset):
        for i in range(len(small_dataset)):
            s = small_dataset[i]
            assert s["prefix_ids"].shape[0] < s["all_ids"].shape[0]

    def test_prefix_starts_at_bos(self, small_dataset):
        s = small_dataset[0]
        assert s["prefix_ids"][0].item() == BOS_ID

    def test_all_ends_at_eos(self, small_dataset):
        s = small_dataset[0]
        assert s["all_ids"][-1].item() == EOS_ID

    def test_positions_are_sequential(self, small_dataset):
        s = small_dataset[0]
        pos = s["prefix_positions"]
        expected = torch.arange(len(pos))
        assert torch.all(pos == expected)

    def test_seq_len_matches_all_ids(self, small_dataset):
        s = small_dataset[0]
        assert s["seq_len"].item() == s["all_ids"].shape[0]

    def test_random_prefix_fraction(self, small_dataset):
        """Different calls to __getitem__ should produce different prefix lengths."""
        lens = set()
        for _ in range(10):
            s = small_dataset[0]
            lens.add(s["prefix_ids"].shape[0])
        # With randomness, we should see at least 2 distinct prefix lengths
        assert len(lens) >= 2

    def test_seeded_dataset_is_deterministic(self, tmp_py_files):
        ds1 = PythonCodeDataset(tmp_py_files, min_seq_len=5, max_seq_len=512, seed=123)
        ds2 = PythonCodeDataset(tmp_py_files, min_seq_len=5, max_seq_len=512, seed=123)
        # Same seed → same prefix lengths on first call
        assert ds1[0]["prefix_ids"].shape[0] == ds2[0]["prefix_ids"].shape[0]


class TestCollateFn:
    def test_output_keys(self, small_dataset):
        batch = collate_fn([small_dataset[i] for i in range(3)])
        required = {"prefix_tokens", "prefix_positions", "prefix_mask",
                    "target_tokens", "target_positions", "target_mask",
                    "seq_len", "prefix_frac"}
        assert required.issubset(batch.keys())

    def test_padded_to_max_len(self, small_dataset):
        samples = [small_dataset[i] for i in range(4)]
        batch = collate_fn(samples)
        max_prefix = max(s["prefix_ids"].shape[0] for s in samples)
        max_all    = max(s["all_ids"].shape[0]    for s in samples)
        assert batch["prefix_tokens"].shape[1] == max_prefix
        assert batch["target_tokens"].shape[1] == max_all

    def test_pad_id_in_padded_positions(self, small_dataset):
        samples = [small_dataset[i] for i in range(4)]
        batch = collate_fn(samples)
        # Padding positions should be PAD_ID
        mask = batch["prefix_mask"]
        tokens = batch["prefix_tokens"]
        padded_positions = ~mask
        if padded_positions.any():
            assert (tokens[padded_positions] == PAD_ID).all()

    def test_mask_matches_valid_tokens(self, small_dataset):
        samples = [small_dataset[i] for i in range(3)]
        batch = collate_fn(samples)
        for i, s in enumerate(samples):
            n = s["prefix_ids"].shape[0]
            assert batch["prefix_mask"][i, :n].all()
            if n < batch["prefix_tokens"].shape[1]:
                assert not batch["prefix_mask"][i, n:].any()

    def test_dataloader_integration(self, small_dataset):
        loader = DataLoader(small_dataset, batch_size=4, collate_fn=collate_fn)
        batch = next(iter(loader))
        assert batch["prefix_tokens"].shape[0] == 4
        assert batch["prefix_tokens"].dtype == torch.long
        assert batch["prefix_mask"].dtype == torch.bool
