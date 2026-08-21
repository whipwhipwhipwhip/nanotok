"""BPE trained from scratch over regex-split chunks.

The split pattern is what stops merges from crossing boundaries you don't want
crossed — without one, " the" and "the." become separate tokens and punctuation
gets welded onto words. The default `(?s).+` is the no-op: one chunk, the whole
text, which is the "basic" tokenizer from the lecture. Pass GPT-4's or GPT-2's
pattern (both available from `TiktokenBPETokenizer`, or `tiktoken`'s `_pat_str`)
for anything realistic.

Special tokens are declared, never learned: they take the ids directly above the
256 bytes, in the order they were declared, and the merges are numbered above
them. So their ids depend on the config and nothing else — an untrained
tokenizer already has them, and retraining on a different corpus leaves them
where they were.
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

# The whole text as one chunk: no splitting, merges free to cross anything.
NO_SPLIT_PATTERN = r"(?s).+"


class RegexTokenizer(ByteTokenizer):
    """Trainable byte-pair encoder. Untrained, it is exactly a `ByteTokenizer`."""

    kind: ClassVar[str] = "regex_bpe"

    def __init__(
        self,
        *,
        special_tokens: list[str] | None = None,
        split_pattern: str = NO_SPLIT_PATTERN,
        pad_vocab_to: int | None = None,
    ) -> None:
        # Keyword-only: `RegexTokenizer(pattern)` used to mean the split pattern,
        # and a positional call that now means "special tokens" would be a
        # silently different tokenizer rather than an error.
        super().__init__(pad_vocab_to=pad_vocab_to)
        self.splitter = split_pattern
        self.merge_map: dict[tuple[int, int], int] = {}
        self.special_tokens = validate_specials(special_tokens or [])
        self._instantiate_vocab()

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Learn merges until the vocabulary reaches `vocab_size`.

        Idempotent by reset rather than by accumulation: training twice on
        different corpora would otherwise produce merges numbered against a
        vocabulary that no longer exists.
        """

        self._instantiate_vocab()

        # The ceiling counts the specials: they occupy ids before the first
        # merge is learned, so a vocab_size of 256 + len(specials) buys nothing.
        #
        # It does not count the padding, and `len(self.vocab)` rather than
        # `self.vocab_size` is what keeps it from doing so. This argument asks
        # for merges to be learned; padding adds ids that no merge can reach, so
        # a padded tokenizer trained to 300 has 300 real ids and reports whatever
        # it was padded to.
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

            # Nothing repeats any more: the corpus is too small (or too varied)
            # to fill the requested vocabulary. Stop short rather than invent
            # merges — `vocab_size` is a ceiling, not a promise.
            if count < 2:
                break

            self.vocab[new_token] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merge_map[pair] = new_token
            groups = merge_groups(groups, pair, new_token)

            if verbose:
                print(f"merge {new_token}: {pair} -> {self.vocab[new_token]!r} ({count}x)")

            new_token += 1

    def _encode_ordinary(self, text: str) -> list[int]:
        # finditer, matching `train`. `findall` returns capture groups rather
        # than whole matches the moment a pattern grows one, so the two would
        # silently disagree about what a chunk is.
        return [
            token
            for match in regex.finditer(self.splitter, text)
            for token in apply_merges(text_to_bytes(match.group()), self.merge_map)
        ]

    def to_dict(self) -> dict:
        # Merges as a list of triples: JSON has no tuple keys, and the pair is
        # the key. Order is the id order, so the file reads as the training log
        # it effectively is.
        return {
            "kind": self.kind,
            "splitter": self.splitter,
            "merges": [
                [p0, p1, idx]
                for (p0, p1), idx in sorted(self.merge_map.items(), key=lambda kv: kv[1])
            ],
            # A list, and never sorted: declaration order *is* the id
            # assignment. The merge ids above are absolute and were numbered
            # above these, so a recipe that dropped them would rebuild a
            # vocabulary with holes at 256.. and a short `vocab_size`.
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
        """The vocabulary before any merges: the bytes, then the specials.

        Called by `train` as well as `__init__`, because training resets. The
        specials have to be restored by that reset and not merely preserved — if
        they were numbered after the merges instead, or re-registered on top of a
        second training run, retraining would renumber them and quietly
        invalidate every checkpoint trained against the first vocabulary.
        """
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_map = {}

        # Directly above the bytes and below the merges, in declaration order:
        # the ids are then a fact about the config alone, so an untrained
        # tokenizer already has them and a retrained one still agrees.
        for offset, special in enumerate(self.special_tokens):
            token = 256 + offset
            self.vocab[token] = special.encode("utf-8")
            self.special_map[special] = token
