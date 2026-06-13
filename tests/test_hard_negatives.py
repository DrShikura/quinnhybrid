"""
Tests for hard negative generation (Phase 2).
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.hard_negatives import HardNegativeGenerator
from data.tokenizer import PythonStructuralTokenizer, BOS_ID, EOS_ID, _VOCAB_INDEX


def test_hard_negative_generator_init():
    gen = HardNegativeGenerator()
    assert gen.swap_count == 2
    assert len(gen.swap_categories) > 0


def test_swap_categories_populated():
    gen = HardNegativeGenerator()
    # Check some key categories exist
    assert "KEYWORD" in gen.swap_categories
    assert "OP_ARITH" in gen.swap_categories
    assert "NAME" in gen.swap_categories


def test_generate_basic():
    gen = HardNegativeGenerator(seed=42)
    tok = PythonStructuralTokenizer()

    # Simple sequence: "x = 1" encoded
    source = "x = 1"
    positive_ids = tok.encode(source)

    # Generate a negative
    negative_ids = gen.generate(positive_ids.copy(), num_swaps=1)

    # Should have same length
    assert len(negative_ids) == len(positive_ids)

    # Should differ in at least one position (with high probability)
    # (might rarely be identical if only swappable with itself)
    assert positive_ids != negative_ids or len([i for i in range(len(positive_ids))
                                                   if gen._get_category(positive_ids[i]) is None]) == len(positive_ids)


def test_generate_no_swappable_tokens():
    gen = HardNegativeGenerator(seed=42)
    # Sequence of only special tokens (BOS, EOS)
    seq = [BOS_ID, EOS_ID]
    negative = gen.generate(seq)
    # Should return unchanged
    assert negative == seq


def test_swap_within_category():
    gen = HardNegativeGenerator(seed=42)
    tok = PythonStructuralTokenizer()

    # Encode a sequence with keywords
    source = "if x: pass"
    positive_ids = tok.encode(source)

    negative_ids = gen.generate(positive_ids.copy(), num_swaps=2)

    # Verify: if any keywords were swapped, they should swap with other keywords
    keyword_ids = gen.swap_categories["KEYWORD"]
    for pos_id, neg_id in zip(positive_ids, negative_ids):
        if pos_id in keyword_ids and pos_id != neg_id:
            # pos_id was swapped, neg_id should also be a keyword
            assert neg_id in keyword_ids


def test_generate_multiple():
    gen = HardNegativeGenerator(seed=42)
    tok = PythonStructuralTokenizer()
    source = "x = 1"
    ids = tok.encode(source)

    negatives = gen.generate_multiple(ids, count=5)
    assert len(negatives) == 5
    assert all(len(n) == len(ids) for n in negatives)


def test_deterministic_with_seed():
    gen1 = HardNegativeGenerator(seed=123)
    gen2 = HardNegativeGenerator(seed=123)

    source = "def foo(): return x"
    tok = PythonStructuralTokenizer()
    ids = tok.encode(source)

    neg1 = gen1.generate(ids.copy())
    neg2 = gen2.generate(ids.copy())

    assert neg1 == neg2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
