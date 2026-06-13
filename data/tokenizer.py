"""
Python structural tokenizer for QuINN Phase 1.

Maps Python source code to a *structural* vocabulary that captures
syntactic roles without encoding specific identifier names. This is what
QuINN learns from: the document argues that structural role (keyword,
bracket, operator) carries more predictive signal about global shape
than surface-level token identity.

Vocabulary design:
  - All keywords mapped individually (def, class, if, for, while, …)
  - Brackets/delimiters mapped individually (critical for nesting structure)
  - Operators collapsed to broader categories (+/-/*, comparison, assignment…)
  - Identifiers → NAME, numbers → NUMBER, strings → STRING
  - Indentation tokens (INDENT/DEDENT) kept explicitly
  - Special: <PAD>, <BOS>, <EOS>, <UNK>
"""

import io
import keyword
import tokenize
from typing import List, Tuple, Optional


# ── Vocabulary ───────────────────────────────────────────────────────────────

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
    "LPAREN", "RPAREN",    # ( )
    "LBRACKET", "RBRACKET", # [ ]
    "LBRACE", "RBRACE",    # { }
    "COLON", "SEMICOLON", "COMMA", "DOT", "ELLIPSIS",
    "AT",                  # decorator @
    "ARROW",               # ->
    "STAR", "DOUBLESTAR",  # * **
]

_OPERATORS = [
    "OP_ASSIGN",           # =
    "OP_AUGASSIGN",        # +=, -=, *=, /=, //=, %=, **=, &=, |=, ^=, >>=, <<=
    "OP_ARITH",            # + - * / // % **
    "OP_COMPARE",          # == != < > <= >=
    "OP_BITWISE",          # & | ^ ~ << >>
    "OP_WALRUS",           # :=
]

_STRUCTURAL = [
    "NAME",                # generic identifier (non-keyword)
    "NUMBER",              # any numeric literal
    "STRING",              # any string/bytes literal
    "FSTRING_START",       # f-string start
    "NEWLINE",             # logical end of statement
    "NL",                  # non-logical newline (inside brackets)
    "INDENT",
    "DEDENT",
    "COMMENT",
    "ENDMARKER",
    "ENCODING",
    "TYPE_COMMENT",
]

VOCAB = _SPECIAL + _KEYWORDS + _DELIMITERS + _OPERATORS + _STRUCTURAL

_VOCAB_INDEX = {tok: i for i, tok in enumerate(VOCAB)}

PAD_ID = _VOCAB_INDEX["<PAD>"]
BOS_ID = _VOCAB_INDEX["<BOS>"]
EOS_ID = _VOCAB_INDEX["<EOS>"]
UNK_ID = _VOCAB_INDEX["<UNK>"]

# Operator → structural category map
_AUGASSIGN_OPS = {"+=", "-=", "*=", "/=", "//=", "%=", "**=", "&=", "|=", "^=", ">>=", "<<=", "@="}
_ARITH_OPS = {"+", "-", "*", "/", "//", "%"}
_COMPARE_OPS = {"==", "!=", "<", ">", "<=", ">="}
_BITWISE_OPS = {"&", "|", "^", "~", "<<", ">>"}


class PythonStructuralTokenizer:
    """
    Tokenizes Python source code into structural token IDs.

    Usage:
        tok = PythonStructuralTokenizer()
        ids = tok.encode(source_code)       # List[int]
        back = tok.decode(ids)              # List[str] – structural token names
    """

    def __init__(self):
        self.vocab = VOCAB
        self.vocab_size = len(VOCAB)
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID

    # ── encoding ─────────────────────────────────────────────────────────

    def encode(self, source: str, add_special: bool = True) -> List[int]:
        """Tokenize source and return list of structural token IDs."""
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        except tokenize.TokenError:
            # Partial files are acceptable; tokenize what we can
            try:
                tokens = self._tokenize_partial(source)
            except Exception:
                return [BOS_ID, EOS_ID] if add_special else []

        ids: List[int] = []
        if add_special:
            ids.append(BOS_ID)

        prev_token_string: Optional[str] = None
        for tok in tokens:
            tok_id = self._map_token(tok, prev_token_string)
            if tok_id is not None:
                ids.append(tok_id)
            prev_token_string = tok.string

        if add_special:
            ids.append(EOS_ID)
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
                pass  # keep previous successful parse
        return all_toks

    def _map_token(self, tok, prev_string: Optional[str]) -> Optional[int]:
        """Map a tokenize.TokenInfo to a structural vocab ID."""
        tt = tok.type
        ts = tok.string

        # Skip ENCODING token (first token in file)
        if tt == tokenize.ENCODING:
            return _VOCAB_INDEX.get("ENCODING")

        if tt == tokenize.ENDMARKER:
            return _VOCAB_INDEX.get("ENDMARKER")

        if tt == tokenize.NAME:
            if keyword.iskeyword(ts) or ts in ("True", "False", "None"):
                # Handle compound operators expressed as adjacent tokens
                if ts == "not":
                    return _VOCAB_INDEX.get("not")
                return _VOCAB_INDEX.get(ts, UNK_ID)
            return _VOCAB_INDEX.get("NAME")

        if tt == tokenize.NUMBER:
            return _VOCAB_INDEX.get("NUMBER")

        if tt in (tokenize.STRING,):
            return _VOCAB_INDEX.get("STRING")

        if tt == tokenize.FSTRING_START if hasattr(tokenize, 'FSTRING_START') else False:
            return _VOCAB_INDEX.get("FSTRING_START")

        if tt == tokenize.NEWLINE:
            return _VOCAB_INDEX.get("NEWLINE")

        if tt == tokenize.NL:
            return _VOCAB_INDEX.get("NL")

        if tt == tokenize.INDENT:
            return _VOCAB_INDEX.get("INDENT")

        if tt == tokenize.DEDENT:
            return _VOCAB_INDEX.get("DEDENT")

        if tt == tokenize.COMMENT:
            return _VOCAB_INDEX.get("COMMENT")

        if tt == tokenize.TYPE_COMMENT if hasattr(tokenize, 'TYPE_COMMENT') else False:
            return _VOCAB_INDEX.get("TYPE_COMMENT")

        if tt == tokenize.OP:
            return self._map_op(ts)

        # Unknown – skip rather than crash
        return None

    def _map_op(self, s: str) -> int:
        if s == "(":   return _VOCAB_INDEX["LPAREN"]
        if s == ")":   return _VOCAB_INDEX["RPAREN"]
        if s == "[":   return _VOCAB_INDEX["LBRACKET"]
        if s == "]":   return _VOCAB_INDEX["RBRACKET"]
        if s == "{":   return _VOCAB_INDEX["LBRACE"]
        if s == "}":   return _VOCAB_INDEX["RBRACE"]
        if s == ":":   return _VOCAB_INDEX["COLON"]
        if s == ";":   return _VOCAB_INDEX["SEMICOLON"]
        if s == ",":   return _VOCAB_INDEX["COMMA"]
        if s == ".":   return _VOCAB_INDEX["DOT"]
        if s == "...": return _VOCAB_INDEX["ELLIPSIS"]
        if s == "@":   return _VOCAB_INDEX["AT"]
        if s == "->":  return _VOCAB_INDEX["ARROW"]
        if s == "*":   return _VOCAB_INDEX["STAR"]
        if s == "**":  return _VOCAB_INDEX["DOUBLESTAR"]
        if s == "=":   return _VOCAB_INDEX["OP_ASSIGN"]
        if s == ":=":  return _VOCAB_INDEX["OP_WALRUS"]
        if s in _AUGASSIGN_OPS: return _VOCAB_INDEX["OP_AUGASSIGN"]
        if s in _ARITH_OPS:     return _VOCAB_INDEX["OP_ARITH"]
        if s in _COMPARE_OPS:   return _VOCAB_INDEX["OP_COMPARE"]
        if s in _BITWISE_OPS:   return _VOCAB_INDEX["OP_BITWISE"]
        return UNK_ID

    # ── decoding ─────────────────────────────────────────────────────────

    def decode(self, ids: List[int]) -> List[str]:
        """Convert list of IDs back to structural token names (for debugging)."""
        return [self.vocab[i] if i < len(self.vocab) else "<UNK>" for i in ids]

    def id_to_name(self, token_id: int) -> str:
        if 0 <= token_id < len(self.vocab):
            return self.vocab[token_id]
        return "<UNK>"
