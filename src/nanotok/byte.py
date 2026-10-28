"""The simplest tokenizer: one id per byte, no merges, vocab 256.

It works fine on its own (it can encode any text with no training), but it's
mainly the base class for the BPE tokenizers. A BPE tokenizer with zero merges
is just this, so the subclasses inherit `decode` and `vocab_size` and only
override `_encode_ordinary`.

Special tokens are handled here too. Reserving an id for `<|endoftext|>` and
cutting it out of the text before merging works the same way for both BPE
tokenizers. The only difference is where the ids come from (`RegexTokenizer`
numbers them after the bytes, `TiktokenBPETokenizer` uses the encoding's own
ids), so subclasses fill in `special_map` and inherit the rest. `ByteTokenizer`
itself has no specials.

Vocabulary padding is here for the same reason: it only changes `vocab_size`,
which works the same for every subclass. See `base.pad_vocab_size`.
"""

from __future__ import annotations

from typing import ClassVar, Sequence

from .base import Tokenizer, pad_record, pad_vocab_size, undecodable_id
from .merges import split_on_specials, text_to_bytes


class ByteTokenizer(Tokenizer):
    """One id per byte. Subclasses grow `vocab` past 256 and override `_encode_ordinary`."""

    kind: ClassVar[str] = "byte"

    def __init__(self, pad_vocab_to: int | None = None) -> None:
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        #: text -> id for special tokens. Empty here; subclasses that support
        #: specials fill it in.
        self.special_map: dict[str, int] = {}
        #: Report this as `vocab_size` instead. This is config rather than
        #: learned state, so `train` keeps it.
        self.pad_vocab_to = pad_vocab_to

    @property
    def unpadded_vocab_size(self) -> int:
        # Computed from `vocab` rather than stored, so it's always up to date,
        # including before and during training.
        #
        # Highest id plus one, not the number of entries, because this sizes an
        # embedding table and gaps still need rows. cl100k's specials start at
        # 100257 and leave 100256 unused, so counting entries would come out one
        # short.
        return max(self.vocab) + 1

    @property
    def vocab_size(self) -> int:
        # The embedding table size, including padding. The merge code uses
        # `unpadded_vocab_size` instead, since new merge ids follow on from the
        # real ids and padding shouldn't move them.
        return pad_vocab_size(self.unpadded_vocab_size, self.pad_vocab_to)

    def encode(self, text: str) -> list[int]:
        """Specials go to their reserved id, everything else through the merges.

        The text is split on specials before any merging, so ordinary text can
        never produce a special's id, and no merge can contain one. One side
        effect: a declared special is always special, so a corpus containing the
        literal text `<|endoftext|>` encodes it as the single special id.
        """
        ids: list[int] = []

        for part in split_on_specials(text, self.special_map):
            special = self.special_map.get(part)
            if special is None:
                ids.extend(self._encode_ordinary(part))
            else:
                ids.append(special)

        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        """Encode text that contains no special tokens."""
        return text_to_bytes(text)

    def decode(self, ids: Sequence[int]) -> str:
        try:
            raw = b"".join(self.vocab[i] for i in ids)
        except KeyError as exc:
            raise undecodable_id(
                exc.args[0], self.unpadded_vocab_size, self.pad_vocab_to
            ) from None

        # errors="replace" because a sampled sequence can end partway through a
        # multi-byte character, and that shouldn't raise.
        return raw.decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        # Only the padding needs recording; the 256 bytes are always the same.
        return {"kind": self.kind, **pad_record(self.pad_vocab_to)}

    @classmethod
    def from_dict(cls, data: dict) -> "ByteTokenizer":
        return cls(pad_vocab_to=data.get("pad_vocab_to"))

    def __eq__(self, other: object) -> bool:
        # Not a dataclass (subclasses have mutable merge state), so equality is
        # defined as "would rebuild the same way".
        return type(other) is type(self) and other.to_dict() == self.to_dict()

    def __repr__(self) -> str:
        # Show the unpadded size too, so a padded vocab_size isn't mistaken for
        # the real one.
        padded = (
            "" if self.pad_vocab_to is None else f" padded from {self.unpadded_vocab_size}"
        )
        return f"{type(self).__name__}(vocab_size={self.vocab_size}{padded})"
