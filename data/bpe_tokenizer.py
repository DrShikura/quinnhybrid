"""
BPE (Byte Pair Encoding) tokenizer for Python code.

Preserves identifier information while keeping structural tokens atomic:
  - Keywords (def, class, if, etc.) stay as single tokens
  - Brackets, delimiters, operators stay as single tokens
  - INDENT, DEDENT, special tokens stay atomic
  - Identifiers, strings, numbers, comments use BPE subword encoding

Target vocab size: 5000-8000 tokens (configurable).

Interface matches PythonStructuralTokenizer:
  encode(source, add_special=True) → List[int]
  decode(ids) → List[str]
  vocab, vocab_size, pad_id, bos_id, eos_id, unk_id properties
  train(corpus_files, vocab_size=5000, save_path=...) classmethod

Design:
  1. Atomic tokens: keywords, brackets, delimiters, operators, INDENT/DEDENT
  2. Subword tokens: identifiers, strings, numbers, comments → BPE
  3. Training: learn merges from corpus using standard BPE algorithm
  4. Inference: apply learned merges during tokenization
"""

import io
import json
import keyword
import tokenize
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict, Counter


# ── Vocabulary constants ─────────────────────────────────────────────────────

_SPECIAL = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

_KEYWORDS = [
    "def", "class", "return", "yield", "yield_from",
    "if", "elif", "else", "for", "while", "break", "continue",
    "try", "except", "finally", "raise", "with", "as",
    "import", "from", "pass", "del", "global", "nonlocal",
    "lambda", "and", "or", "not", "in", "is", "is_not", "not_in",
    "True", "False", "None", "async", "await",
]

_DELIMITERS = [
    "LPAREN", "RPAREN",
    "LBRACKET", "RBRACKET",
    "LBRACE", "RBRACE",
    "COLON", "SEMICOLON", "COMMA", "DOT", "ELLIPSIS",
    "AT", "ARROW",
    "STAR", "DOUBLESTAR",
]

_OPERATORS = [
    "OP_ASSIGN",
    "OP_AUGASSIGN",
    "OP_ARITH",
    "OP_COMPARE",
    "OP_BITWISE",
    "OP_WALRUS",
]

_STRUCTURAL = [
    "NAME",
    "NAME_CLASS",
    "NAME_FUNC",
    "NAME_ATTR",
    "NAME_CALL",
    "NAME_PARAM",
    "NAME_ASSIGN",
    "NAME_OTHER",
    "NUMBER",
    "STRING",
    "FSTRING_START",
    "NEWLINE",
    "NL",
    "INDENT",
    "DEDENT",
    "COMMENT",
    "ENDMARKER",
    "ENCODING",
    "TYPE_COMMENT",
]

# These tokens are NEVER merged in BPE (kept atomic)
_ATOMIC_STRUCTURAL = _SPECIAL + _KEYWORDS + _DELIMITERS + _OPERATORS + _STRUCTURAL

# Operator mappings (from original tokenizer)
_AUGASSIGN_OPS = {"+=", "-=", "*=", "/=", "//=", "%=", "**=", "&=", "|=", "^=", ">>=", "<<=", "@="}
_ARITH_OPS = {"+", "-", "*", "/", "//", "%"}
_COMPARE_OPS = {"==", "!=", "<", ">", "<=", ">="}
_BITWISE_OPS = {"&", "|", "^", "~", "<<", ">>"}


class BytePairTokenizer:
    """
    BPE tokenizer for Python code. Preserves identifiers via subword encoding
    while keeping structural tokens (keywords, brackets, operators, etc.) atomic.

    Usage:
        # Training (one-time):
        tok = BytePairTokenizer.train(['file1.py', 'file2.py', ...],
                                      vocab_size=5000,
                                      save_path='data/bpe_vocab')

        # Inference (load trained tokenizer):
        tok = BytePairTokenizer('data/bpe_vocab')
        ids = tok.encode(source_code)
        tokens = tok.decode(ids)
    """

    def __init__(self, vocab_path: Optional[str] = None):
        """
        Initialize tokenizer.

        Args:
            vocab_path: Path to directory containing vocab.json and merges.txt.
                       If None, uses default structural vocab (no BPE).
        """
        self.vocab_path = Path(vocab_path) if vocab_path else None
        self.merges: Dict[Tuple[str, str], int] = {}  # (token1, token2) → merge rank
        self.vocab: List[str] = []
        self.vocab_index: Dict[str, int] = {}

        if vocab_path and Path(vocab_path).exists():
            self._load_vocab(Path(vocab_path))
        else:
            # Fallback to structural vocabulary (no merges, no BPE)
            self._init_structural_vocab()

    # ── Initialization ───────────────────────────────────────────────────────

    def _init_structural_vocab(self):
        """Initialize with structural vocab only (no BPE merges)."""
        self.vocab = _ATOMIC_STRUCTURAL
        self.vocab_index = {tok: i for i, tok in enumerate(self.vocab)}
        self.merges = {}

    def _load_vocab(self, vocab_path: Path):
        """Load vocab.json and merges.txt from disk."""
        vocab_file = vocab_path / "vocab.json"
        merges_file = vocab_path / "merges.txt"

        if not vocab_file.exists() or not merges_file.exists():
            raise FileNotFoundError(
                f"Missing vocab files in {vocab_path}. "
                f"Expected: {vocab_file} and {merges_file}"
            )

        # Load vocabulary
        with open(vocab_file) as f:
            self.vocab_index = json.load(f)
        self.vocab = [""] * len(self.vocab_index)
        for token, idx in self.vocab_index.items():
            self.vocab[idx] = token

        # Load merges (ordered list of merge operations)
        self.merges = {}
        with open(merges_file) as f:
            for rank, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) == 2:
                    self.merges[tuple(parts)] = rank

    def _save_vocab(self, vocab_dict: Dict[str, int], merges_list: List[Tuple[str, str]], save_path: Path):
        """Save vocab.json and merges.txt to disk."""
        save_path.mkdir(parents=True, exist_ok=True)

        # Save vocab
        vocab_file = save_path / "vocab.json"
        with open(vocab_file, 'w') as f:
            json.dump(vocab_dict, f, indent=2)

        # Save merges
        merges_file = save_path / "merges.txt"
        with open(merges_file, 'w') as f:
            for t1, t2 in merges_list:
                f.write(f"{t1} {t2}\n")

        print(f"  Saved vocab to: {vocab_file}")
        print(f"  Saved merges to: {merges_file}")

    # ── Training (one-time, corpus-level) ────────────────────────────────────

    @classmethod
    def train(
        cls,
        corpus_files: List[str],
        vocab_size: int = 5000,
        save_path: Optional[str] = None,
    ) -> 'BytePairTokenizer':
        """
        Train BPE tokenizer on a corpus of Python files.

        Args:
            corpus_files: List of .py file paths or a .txt file with one path per line
            vocab_size: Target vocabulary size (5000-8000 recommended)
            save_path: Where to save vocab.json and merges.txt.
                      If None, uses './bpe_vocab'.

        Returns:
            Initialized BytePairTokenizer with trained vocabulary.
        """
        if save_path is None:
            save_path = 'data/bpe_vocab'
        save_path = Path(save_path)

        print(f"Training BPE tokenizer on corpus...")
        print(f"  Target vocab size: {vocab_size}")

        # Load corpus files
        files_to_tokenize = cls._load_file_list(corpus_files)
        print(f"  Found {len(files_to_tokenize)} Python files")

        # Phase 1: Collect all tokens from corpus, pre-split into "words"
        word_frequencies = cls._collect_words(files_to_tokenize)
        print(f"  Collected {len(word_frequencies)} unique words")

        # Phase 2: Learn BPE merges
        print(f"  Learning BPE merges...")
        merges, final_vocab = cls._learn_bpe(
            word_frequencies,
            vocab_size,
            initial_vocab=_ATOMIC_STRUCTURAL,
        )

        # Phase 3: Save vocabulary
        vocab_dict = {token: i for i, token in enumerate(final_vocab)}
        cls(None)._save_vocab(vocab_dict, merges, save_path)

        # Phase 4: Return initialized tokenizer
        tok = cls(str(save_path))
        print(f"  Final vocab size: {len(tok.vocab)}")
        return tok

    @staticmethod
    def _load_file_list(source: List[str]) -> List[Path]:
        """Load list of file paths from corpus_files."""
        if isinstance(source, str):
            source = [source]

        files = []
        for item in source:
            p = Path(item)
            if p.is_file() and item.endswith('.txt'):
                # Read from file list
                lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
                files.extend([Path(ln.strip()) for ln in lines if ln.strip()])
            elif p.is_dir():
                # Recursively find .py files
                files.extend(sorted(p.rglob('*.py')))
            elif p.is_file():
                files.append(p)

        return [f for f in files if f.exists()]

    @staticmethod
    def _collect_words(file_paths: List[Path]) -> Dict[str, int]:
        """
        Tokenize all files and collect word frequencies.

        A "word" is a sequence of tokens from one Python token (identifier,
        string, number, or comment). Atomic tokens are kept as-is.

        Returns:
            Dict[word, frequency] where word is a space-separated char sequence.
        """
        word_freqs = defaultdict(int)

        for i, file_path in enumerate(file_paths):
            if (i + 1) % 500 == 0:
                print(f"    Processed {i + 1} files...")

            try:
                source = file_path.read_text(encoding='utf-8', errors='replace')
                tokenizer = BytePairTokenizer(None)  # Use structural tokenizer to collect tokens
                tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))

                for tok in tokens:
                    tt = tok.type
                    ts = tok.string

                    # Skip empty/whitespace tokens
                    if not ts.strip() or ts in ('\n', '\t', ' '):
                        continue

                    # Map token to word (for BPE learning)
                    word = BytePairTokenizer._token_to_word(tok, tt, ts)
                    if word:
                        word_freqs[word] += 1
            except Exception:
                pass

        return dict(word_freqs)

    @staticmethod
    def _token_to_word(tok, tt: int, ts: str) -> Optional[str]:
        """
        Convert a tokenize.TokenInfo to a "word" for BPE learning.

        Atomic tokens (keywords, brackets, etc.) → single token word.
        Subword-able tokens (identifier, string, number) → char-sequence word.

        Returns:
            Space-separated string of chars/tokens for BPE, or None if skip.
        """
        # Skip encoding/endmarker
        if tt in (tokenize.ENCODING, tokenize.ENDMARKER):
            return None

        # Keywords and built-ins → atomic
        if tt == tokenize.NAME:
            if keyword.iskeyword(ts) or ts in ("True", "False", "None"):
                return ts
            # Non-keyword identifier → split into chars for BPE
            return " ".join(ts) + " </w>"

        # Structural tokens → atomic
        if tt == tokenize.INDENT:
            return "INDENT"
        if tt == tokenize.DEDENT:
            return "DEDENT"
        if tt == tokenize.NEWLINE:
            return "NEWLINE"
        if tt == tokenize.NL:
            return "NL"
        if tt == tokenize.COMMENT:
            # Split comment text into chars for BPE
            return " ".join(ts) + " </w>"
        if tt == tokenize.TYPE_COMMENT:
            return "TYPE_COMMENT"

        # Numbers and strings → split into chars
        if tt == tokenize.NUMBER:
            return " ".join(ts) + " </w>"
        if tt == tokenize.STRING:
            # Don't tokenize string contents (too long), just mark as STRING
            return "STRING"
        if hasattr(tokenize, 'FSTRING_START') and tt == tokenize.FSTRING_START:
            return "FSTRING_START"

        # Operators → map to structural names (like original tokenizer)
        if tt == tokenize.OP:
            op_name = BytePairTokenizer._map_op(ts)
            return op_name if op_name else None

        return None

    @staticmethod
    def _map_op(s: str) -> Optional[str]:
        """Map operator symbol to structural token name."""
        ops = {
            "(": "LPAREN", ")": "RPAREN",
            "[": "LBRACKET", "]": "RBRACKET",
            "{": "LBRACE", "}": "RBRACE",
            ":": "COLON", ";": "SEMICOLON", ",": "COMMA",
            ".": "DOT", "...": "ELLIPSIS",
            "@": "AT", "->": "ARROW",
            "*": "STAR", "**": "DOUBLESTAR",
            "=": "OP_ASSIGN",
            ":=": "OP_WALRUS",
        }
        if s in ops:
            return ops[s]
        if s in _AUGASSIGN_OPS:
            return "OP_AUGASSIGN"
        if s in _ARITH_OPS:
            return "OP_ARITH"
        if s in _COMPARE_OPS:
            return "OP_COMPARE"
        if s in _BITWISE_OPS:
            return "OP_BITWISE"
        return None

    @staticmethod
    def _learn_bpe(
        word_frequencies: Dict[str, int],
        target_vocab_size: int,
        initial_vocab: List[str],
    ) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Learn BPE merges using greedy frequency-based merging.

        Algorithm:
        1. Start with character vocabulary + atomic tokens
        2. Repeatedly:
           a. Count all adjacent token pairs in corpus
           b. Find most frequent pair
           c. Merge in all word sequences
           d. Add to vocabulary
           e. Repeat until target vocab_size

        Args:
            word_frequencies: Dict[word (space-separated), frequency]
            target_vocab_size: Target vocabulary size
            initial_vocab: List of atomic tokens to preserve

        Returns:
            (merges, final_vocab) where merges is list of (token, token) pairs
                                   and final_vocab is full vocabulary list
        """
        # Initialize vocabulary with atomic tokens + characters
        vocab = set(initial_vocab)
        for word in word_frequencies:
            tokens = word.split()
            vocab.update(tokens)

        print(f"    Initial vocab size: {len(vocab)}")

        # Initialize word sequences (words split by spaces + frequencies)
        word_tokenizations: Dict[str, Tuple[List[str], int]] = {}
        for word, freq in word_frequencies.items():
            tokens = tuple(word.split())
            word_tokenizations[word] = (list(tokens), freq)

        merges = []

        # Greedy BPE: repeatedly merge most frequent pair
        iteration = 0
        while len(vocab) < target_vocab_size:
            iteration += 1
            if iteration % 100 == 0:
                print(f"    Iteration {iteration}: vocab size = {len(vocab)}")

            # Count all adjacent pairs
            pair_frequencies = defaultdict(int)
            for word, (tokens, freq) in word_tokenizations.items():
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_frequencies[pair] += freq

            if not pair_frequencies:
                break

            # Find most frequent pair
            best_pair = max(pair_frequencies, key=pair_frequencies.get)
            best_freq = pair_frequencies[best_pair]

            # Merge this pair in all words
            new_token = best_pair[0] + "##" + best_pair[1]
            vocab.add(new_token)
            merges.append(best_pair)

            # Update word tokenizations
            for word, (tokens, freq) in word_tokenizations.items():
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                word_tokenizations[word] = (new_tokens, freq)

        # Convert vocab to sorted list
        final_vocab = _ATOMIC_STRUCTURAL + sorted(list(vocab - set(_ATOMIC_STRUCTURAL)))

        return merges, final_vocab

    # ── Encoding (inference) ─────────────────────────────────────────────────

    def encode(self, source: str, add_special: bool = True) -> List[int]:
        """
        Tokenize Python source and return list of token IDs.

        Args:
            source: Python source code string
            add_special: Whether to add <BOS> and <EOS> tokens

        Returns:
            List of token IDs
        """
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        except tokenize.TokenError:
            try:
                tokens = self._tokenize_partial(source)
            except Exception:
                return [self.bos_id, self.eos_id] if add_special else []

        ids: List[int] = []
        if add_special:
            ids.append(self.bos_id)

        for i, tok in enumerate(tokens):
            prev_tok = tokens[i - 1] if i > 0 else None
            next_tok = tokens[i + 1] if i < len(tokens) - 1 else None
            tok_id = self._map_token_with_context(tok, prev_tok, next_tok)
            if tok_id is not None:
                ids.append(tok_id)

        if add_special:
            ids.append(self.eos_id)

        return ids

    def _tokenize_partial(self, source: str) -> list:
        """Handle TokenError by tokenizing line-by-line."""
        all_toks = []
        lines = source.splitlines(keepends=True)
        buf = ""
        for line in lines:
            buf += line
            try:
                toks = list(tokenize.generate_tokens(io.StringIO(buf).readline))
                all_toks = toks
            except tokenize.TokenError:
                pass
        return all_toks

    def _map_token_with_context(self, tok, prev_tok, next_tok) -> Optional[int]:
        """
        Map a tokenize.TokenInfo to a token ID.

        For atomic tokens (keywords, brackets, operators): direct lookup.
        For subword-able tokens (identifiers, strings, numbers): apply BPE.
        """
        tt = tok.type
        ts = tok.string

        # Special tokens
        if tt == tokenize.ENCODING:
            return self._lookup("ENCODING")
        if tt == tokenize.ENDMARKER:
            return self._lookup("ENDMARKER")

        # Keywords and built-ins
        if tt == tokenize.NAME:
            if keyword.iskeyword(ts) or ts in ("True", "False", "None"):
                return self._lookup(ts)
            # Non-keyword identifier: apply BPE
            return self._encode_subword(ts)

        # Structural tokens
        if tt == tokenize.NUMBER:
            return self._encode_subword(ts)
        if tt == tokenize.STRING:
            return self._lookup("STRING")
        if hasattr(tokenize, 'FSTRING_START') and tt == tokenize.FSTRING_START:
            return self._lookup("FSTRING_START")
        if tt == tokenize.NEWLINE:
            return self._lookup("NEWLINE")
        if tt == tokenize.NL:
            return self._lookup("NL")
        if tt == tokenize.INDENT:
            return self._lookup("INDENT")
        if tt == tokenize.DEDENT:
            return self._lookup("DEDENT")
        if tt == tokenize.COMMENT:
            return self._encode_subword(ts)
        if hasattr(tokenize, 'TYPE_COMMENT') and tt == tokenize.TYPE_COMMENT:
            return self._lookup("TYPE_COMMENT")

        # Operators
        if tt == tokenize.OP:
            op_name = self._map_op(ts)
            return self._lookup(op_name) if op_name else None

        return None

    def _encode_subword(self, text: str) -> Optional[int]:
        """
        Apply BPE merges to text and return single token ID.

        For subword-able tokens (identifiers, numbers, comments):
        1. Split into characters
        2. Apply learned merges iteratively
        3. Return first resulting token ID (or merged if still multiple)

        NOTE: This is a simplification. In practice, we'd want to handle
        multiple subword tokens per identifier. For now, we treat the entire
        identifier as a single merged unit.
        """
        if not text:
            return self._lookup("<UNK>")

        # Split into chars
        tokens = list(text) + ["</w>"]

        # Apply merges
        for merge_pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == merge_pair:
                    merged = tokens[i] + "##" + tokens[i + 1]
                    tokens = tokens[:i] + [merged] + tokens[i + 2:]
                    i += 1
                else:
                    i += 1

        # Join all tokens together (treating as a single "word" token)
        # This is a pragmatic simplification: full BPE would return multiple IDs
        merged_word = "".join(tokens).replace("</w>", "")
        return self._lookup(merged_word)

    def _lookup(self, token: str) -> Optional[int]:
        """Look up token ID, return UNK_ID if not found."""
        return self.vocab_index.get(token, self.vocab_index.get("<UNK>"))

    # ── Decoding ─────────────────────────────────────────────────────────────

    def decode(self, ids: List[int]) -> List[str]:
        """Convert list of token IDs back to token names."""
        tokens = []
        for token_id in ids:
            if 0 <= token_id < len(self.vocab):
                tokens.append(self.vocab[token_id])
            else:
                tokens.append("<UNK>")
        return tokens

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def pad_id(self) -> int:
        return self.vocab_index.get("<PAD>", 0)

    @property
    def bos_id(self) -> int:
        return self.vocab_index.get("<BOS>", 1)

    @property
    def eos_id(self) -> int:
        return self.vocab_index.get("<EOS>", 2)

    @property
    def unk_id(self) -> int:
        return self.vocab_index.get("<UNK>", 3)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def vocab_hash(self) -> str:
        """Hash of vocabulary for cache invalidation."""
        if not self.vocab_path or not self.vocab_path.exists():
            # Structural vocab: use fixed hash
            return hashlib.md5(b"structural-vocab-81").hexdigest()[:16]

        # Hash the vocab and merges files
        h = hashlib.md5()
        for file in [self.vocab_path / "vocab.json", self.vocab_path / "merges.txt"]:
            if file.exists():
                h.update(file.read_bytes())
        return h.hexdigest()[:16]


# Backward compatibility: export as alias for any code expecting the old name
PythonStructuralTokenizer = BytePairTokenizer
