"""Tests for data.tokenizer module."""

import pytest
from data.tokenizer import PythonStructuralTokenizer, VOCAB, PAD_ID, BOS_ID, EOS_ID


SIMPLE_PY = """
def hello(name):
    if name:
        return f"Hello, {name}!"
    return None
"""

COMPLEX_PY = """
class MyClass:
    def __init__(self, x: int = 0):
        self.x = x

    def compute(self, n: int) -> list:
        result = [i ** 2 for i in range(n)]
        return result

    @property
    def value(self):
        return self.x
"""


class TestPythonStructuralTokenizer:
    def setup_method(self):
        self.tok = PythonStructuralTokenizer()

    def test_vocab_size(self):
        assert self.tok.vocab_size == len(VOCAB)
        assert self.tok.vocab_size > 50  # reasonable lower bound

    def test_special_tokens(self):
        assert self.tok.pad_id == PAD_ID
        assert self.tok.bos_id == BOS_ID
        assert self.tok.eos_id == EOS_ID

    def test_encode_returns_bos_eos(self):
        ids = self.tok.encode(SIMPLE_PY, add_special=True)
        assert ids[0] == BOS_ID
        assert ids[-1] == EOS_ID

    def test_encode_no_special(self):
        ids = self.tok.encode(SIMPLE_PY, add_special=False)
        assert ids[0] != BOS_ID
        assert ids[-1] != EOS_ID

    def test_all_ids_in_range(self):
        ids = self.tok.encode(COMPLEX_PY)
        for i in ids:
            assert 0 <= i < self.tok.vocab_size, f"Out-of-range ID: {i}"

    def test_keywords_encoded_individually(self):
        ids = self.tok.encode("def class return if else for while", add_special=False)
        names = self.tok.decode(ids)
        # All known keywords should be in vocabulary as individual entries
        for kw in ("def", "class", "return", "if", "else", "for", "while"):
            assert kw in VOCAB

    def test_brackets_encoded(self):
        ids = self.tok.encode("()[]{}", add_special=False)
        names = self.tok.decode(ids)
        expected = {"LPAREN", "RPAREN", "LBRACKET", "RBRACKET", "LBRACE", "RBRACE"}
        found = set(names)
        assert expected.issubset(found), f"Missing bracket tokens: {expected - found}"

    def test_non_keyword_name_becomes_NAME(self):
        ids = self.tok.encode("foo bar baz", add_special=False)
        names = self.tok.decode(ids)
        # Identifiers become NAME; tokenize also emits NEWLINE/ENDMARKER for the line
        non_structural = {"<BOS>", "<EOS>", "<PAD>", "<UNK>", "NEWLINE", "NL", "ENDMARKER", "ENCODING"}
        identifier_names = [n for n in names if n not in non_structural]
        assert all(n == "NAME" for n in identifier_names), f"Non-NAME tokens: {identifier_names}"
        assert len(identifier_names) == 3  # foo, bar, baz

    def test_decode_roundtrip(self):
        ids = self.tok.encode(SIMPLE_PY)
        names = self.tok.decode(ids)
        assert all(isinstance(n, str) for n in names)
        assert len(names) == len(ids)

    def test_empty_string(self):
        ids = self.tok.encode("", add_special=True)
        assert ids[0] == BOS_ID
        assert ids[-1] == EOS_ID

    def test_partial_file_handled(self):
        """Files with syntax errors (partial code) should not raise."""
        partial = "def foo(\n    x, y"
        ids = self.tok.encode(partial)
        assert len(ids) >= 2

    def test_longer_file_produces_more_tokens(self):
        ids_simple  = self.tok.encode(SIMPLE_PY)
        ids_complex = self.tok.encode(COMPLEX_PY)
        assert len(ids_complex) > len(ids_simple)
