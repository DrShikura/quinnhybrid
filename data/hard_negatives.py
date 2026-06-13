"""
Hard negative generation for Phase 2 contrastive training.

Strategy: Generate structurally-plausible but semantically-wrong sequences by swapping
tokens within the same structural category (e.g., NAME ↔ NAME, OP_ARITH ↔ OP_ARITH).

This tests whether QuINN's waveform captures *structural* invariants: swapping keywords
changes structure, but swapping identifiers doesn't — QuINN should distinguish these.
"""

import random
from typing import List, Dict, Set
from .tokenizer import VOCAB, _KEYWORDS, _DELIMITERS, _OPERATORS, _STRUCTURAL, _VOCAB_INDEX


class HardNegativeGenerator:
    """Generate hard negatives by swapping tokens within structural categories."""

    def __init__(self, seed: int = 42, swap_count: int = 2):
        """
        Args:
            seed: RNG seed
            swap_count: number of positions to swap (1-3 typically)
        """
        self.rng = random.Random(seed)
        self.swap_count = swap_count
        self._build_swap_categories()

    def _build_swap_categories(self):
        """Build groups of token IDs that can swap with each other."""
        self.swap_categories: Dict[str, Set[int]] = {}

        # Keywords: can swap with any other keyword
        self.swap_categories["KEYWORD"] = {
            _VOCAB_INDEX[kw] for kw in _KEYWORDS
        }

        # Delimiters: same delimiter type (parens ↔ parens, brackets ↔ brackets)
        self.swap_categories["LPAREN_RPAREN"] = {
            _VOCAB_INDEX["LPAREN"],
            _VOCAB_INDEX["RPAREN"],
        }
        self.swap_categories["LBRACKET_RBRACKET"] = {
            _VOCAB_INDEX["LBRACKET"],
            _VOCAB_INDEX["RBRACKET"],
        }
        self.swap_categories["LBRACE_RBRACE"] = {
            _VOCAB_INDEX["LBRACE"],
            _VOCAB_INDEX["RBRACE"],
        }
        self.swap_categories["OTHER_DELIM"] = {
            _VOCAB_INDEX[d] for d in _DELIMITERS
            if d not in ["LPAREN", "RPAREN", "LBRACKET", "RBRACKET", "LBRACE", "RBRACE"]
        }

        # Operators: same operator category
        self.swap_categories["OP_ARITH"] = {
            _VOCAB_INDEX["OP_ARITH"],  # +, -, *, /, //, %
        }
        self.swap_categories["OP_COMPARE"] = {
            _VOCAB_INDEX["OP_COMPARE"],  # ==, !=, <, >, <=, >=
        }
        self.swap_categories["OP_ASSIGN"] = {
            _VOCAB_INDEX["OP_ASSIGN"],
            _VOCAB_INDEX["OP_AUGASSIGN"],  # Allow assignment ↔ augmented assignment
        }
        self.swap_categories["OP_BITWISE"] = {
            _VOCAB_INDEX["OP_BITWISE"],  # &, |, ^, ~, <<, >>
        }

        # Structural: swap same structural tokens (not across categories)
        self.swap_categories["NAME"] = {_VOCAB_INDEX["NAME"]}
        self.swap_categories["NUMBER"] = {_VOCAB_INDEX["NUMBER"]}
        self.swap_categories["STRING"] = {_VOCAB_INDEX["STRING"]}
        self.swap_categories["NEWLINE"] = {
            _VOCAB_INDEX["NEWLINE"],
            _VOCAB_INDEX["NL"],
        }
        self.swap_categories["INDENT_DEDENT"] = {
            _VOCAB_INDEX["INDENT"],
            _VOCAB_INDEX["DEDENT"],
        }

    def _get_category(self, token_id: int) -> str:
        """Find which swap category a token belongs to."""
        for cat_name, cat_ids in self.swap_categories.items():
            if token_id in cat_ids:
                return cat_name
        return None  # Token cannot be swapped (special, padding, etc.)

    def generate(self, token_ids: List[int], num_swaps: int = None) -> List[int]:
        """
        Generate a hard negative by swapping 1-k tokens with alternatives in the same category.

        Args:
            token_ids: original sequence (including BOS/EOS if present)
            num_swaps: number of positions to swap (default: self.swap_count)

        Returns:
            modified sequence with swaps
        """
        if num_swaps is None:
            num_swaps = self.rng.randint(1, self.swap_count)

        negative = token_ids.copy()

        # Find positions that can be swapped
        swappable_positions = []
        for i, tok_id in enumerate(negative):
            cat = self._get_category(tok_id)
            if cat is not None and len(self.swap_categories[cat]) > 1:
                swappable_positions.append(i)

        # Sample positions to swap
        if len(swappable_positions) < num_swaps:
            num_swaps = len(swappable_positions)

        if num_swaps == 0:
            return negative  # Can't generate a negative

        positions_to_swap = self.rng.sample(swappable_positions, num_swaps)

        # Perform swaps
        for pos in positions_to_swap:
            tok_id = negative[pos]
            cat = self._get_category(tok_id)
            alternatives = list(self.swap_categories[cat])
            # Remove current token from alternatives
            alternatives = [a for a in alternatives if a != tok_id]
            if alternatives:
                negative[pos] = self.rng.choice(alternatives)

        return negative

    def generate_multiple(
        self, token_ids: List[int], count: int = 1
    ) -> List[List[int]]:
        """Generate multiple hard negatives for the same sequence."""
        return [self.generate(token_ids) for _ in range(count)]
