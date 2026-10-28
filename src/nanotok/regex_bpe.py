"""BPE trained from scratch, over regex-split chunks.

The split pattern stops merges from crossing boundaries you don't want them to.
Without it, punctuation and spaces get merged onto words, so e.g. " the" and
"the." end up as separate tokens. The default `(?s).+` doesn't split at all (the
whole text is one chunk), which is the "basic" tokenizer from the lecture. For
anything realistic, pass GPT-2's or GPT-4's pattern (tiktoken's `_pat_str`, or
`TiktokenBPETokenizer(...).splitter`).

Special tokens are declared, never learned. They get the ids right after the
256 bytes, in the order given, and the merges are numbered after them. So their
ids only depend on the config: an untrained tokenizer already has them, and
retraining on a different corpus doesn't move them.
"""

from __future__ import annotations

from typing import ClassVar

import regex

from .base import pad_record
from .byte import ByteTokenizer
from .merges import (
    apply_merges,
    build_vocab,
    find_largest,
    merge_groups,
    text_to_bytes,
    validate_specials,
)

# The whole text as one chunk: no splitting, merges can cross anything.
NO_SPLIT_PATTERN = r"(?s).+"


class RegexTokenizer(ByteTokenizer):
    """Trainable BPE tokenizer. Untrained, it behaves exactly like `ByteTokenizer`."""

    kind: ClassVar[str] = "regex_bpe"

    def __init__(
        self,
        *,
        special_tokens: list[str] | None = None,
        split_pattern: str = NO_SPLIT_PATTERN,
        pad_vocab_to: int | None = None,
    ) -> None:
        # Keyword-only because the first positional argument used to be the
        # split pattern, and old calls like `RegexTokenizer(pattern)` should
        # fail rather than treat the pattern as special tokens.
        super().__init__(pad_vocab_to=pad_vocab_to)
        self.splitter = split_pattern
        self.merge_map: dict[tuple[int, int], int] = {}
        self.special_tokens = validate_specials(special_tokens or [])
        self._instantiate_vocab()

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Learn merges until the vocabulary reaches `vocab_size`.

        Starts from scratch each time, so training twice gives the result of the
        second run, not the two combined.
        """

        self._instantiate_vocab()

        # `vocab_size` includes the specials, since they take ids before any
        # merges, so 256 + len(specials) would mean no merges at all.
        #
        # It doesn't include padding, which is why this uses `len(self.vocab)`
        # rather than `self.vocab_size`. A padded tokenizer trained to 300 has
        # 300 real ids, and reports the padded size as its `vocab_size`.
        if vocab_size <= len(self.vocab):
            raise ValueError(
                f"vocab_size must exceed the {len(self.vocab)} ids already "
                f"reserved (256 bytes + {len(self.special_tokens)} special "
                f"tokens), got {vocab_size}"
            )

        self.merge_map = {}
        groups = [text_to_bytes(m.group()) for m in regex.finditer(self.splitter, text)]
        new_token = len(self.vocab)

        while new_token < vocab_size:
            count, pair = find_largest(groups)

            # No pair occurs more than once, so the corpus is too small to reach
            # the requested size. Stop early; `vocab_size` is an upper limit.
            if count < 2:
                break

            self.vocab[new_token] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merge_map[pair] = new_token
            groups = merge_groups(groups, pair, new_token)

            if verbose:
                print(f"merge {new_token}: {pair} -> {self.vocab[new_token]!r} ({count}x)")

            new_token += 1

    def _encode_ordinary(self, text: str) -> list[int]:
        # finditer, same as `train`. `findall` returns capture groups instead
        # of whole matches if the pattern has any, so the two could end up
        # splitting text differently.
        return [
            token
            for match in regex.finditer(self.splitter, text)
            for token in apply_merges(text_to_bytes(match.group()), self.merge_map)
        ]

    def to_dict(self) -> dict:
        # Merges as a list of [p0, p1, id], since JSON can't have tuple keys.
        # Sorted by id, so they're in the order they were learned.
        return {
            "kind": self.kind,
            "splitter": self.splitter,
            "merges": [
                [p0, p1, idx]
                for (p0, p1), idx in sorted(self.merge_map.items(), key=lambda kv: kv[1])
            ],
            # A list in declaration order, never sorted, because the order
            # decides the ids. The merge ids above are numbered after these, so
            # without them the rebuilt vocabulary would have a gap at 256.
            "special_tokens": list(self.special_tokens),
            **pad_record(self.pad_vocab_to),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RegexTokenizer":
        tok = cls(
            special_tokens=data["special_tokens"],
            split_pattern=data["splitter"],
            pad_vocab_to=data.get("pad_vocab_to"),
        )
        tok.merge_map = {(p0, p1): idx for p0, p1, idx in data["merges"]}
        tok.vocab = build_vocab(tok.vocab, tok.merge_map)
        return tok

    def _instantiate_vocab(self) -> None:
        """Set up the vocabulary before any merges: the bytes, then the specials.

        `train` calls this too, to reset. The specials are re-added here with
        the same ids each time, so retraining never renumbers them (which would
        break any checkpoint trained with the old ids).
        """
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_map = {}

        # Right after the bytes and before the merges, in declaration order.
        for offset, special in enumerate(self.special_tokens):
            token = 256 + offset
            self.vocab[token] = special.encode("utf-8")
            self.special_map[special] = token
