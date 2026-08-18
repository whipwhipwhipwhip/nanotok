"""The tokenizer that does nothing: one id per byte, no merges, vocab 256.

Useful on its own (it can encode any text ever written, with no training and no
vocabulary to persist), but mostly useful as the base of everything BPE. A BPE
tokenizer with zero merges *is* this, so the subclasses inherit `decode` and
`vocab_size` and override only `_encode_ordinary` — which is why fixing `vocab`
here fixed the bug where the old `GPT4Tokenizer.decode` looked up merged ids in a
table that only ever held the 256 raw bytes.

**Special tokens live here too**, for the same reason: reserving an id for
`<|endoftext|>` and cutting it out of the text before merging is identical work
whether the merges were trained here or came from tiktoken. What differs is only
where the ids come from — `RegexTokenizer` numbers them above the bytes,
`TiktokenBPETokenizer` takes the numbers the encoding already assigned — so
subclasses fill in `special_map` and inherit the rest. `ByteTokenizer` itself
declares none, which makes its `encode` the same one-chunk pass it always was.
"""

from __future__ import annotations

from typing import ClassVar, Sequence

from .base import Tokenizer
from .merges import split_on_specials, text_to_bytes


class ByteTokenizer(Tokenizer):
    """Byte-level identity. Subclasses grow `vocab` past 256 and override `_encode_ordinary`."""

    kind: ClassVar[str] = "byte"

    def __init__(self) -> None:
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        #: text -> id, for ids that are declared rather than merged into
        #: existence. Empty here; subclasses that accept specials fill it.
        self.special_map: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        # Derived from the table rather than stored, so it's right before, during
        # and after training. The old RegexTokenizer set `self.vocab_size` inside
        # `train`, which left an untrained instance with no attribute at all.
        #
        # Highest id plus one, not the number of entries: this sizes an embedding
        # table, so a gap in the id space still has to be paid for. cl100k's
        # specials start at 100257 and leave 100256 unused, so counting entries
        # would hand the model a table one row short of the ids it must embed.
        return max(self.vocab) + 1

    def encode(self, text: str) -> list[int]:
        """Specials to their reserved id, everything else through the merges.

        The split happens before any merging, so a special is unreachable from
        the merge machinery in both directions: ordinary text can't forge its id,
        and no merge can be built out of it. Note the consequence — a declared
        special is *always* special, so a corpus containing the literal text
        `<|endoftext|>` encodes as that one id rather than as its characters.
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
        """Encode a stretch of text known to contain no special tokens."""
        return text_to_bytes(text)

    def decode(self, ids: Sequence[int]) -> str:
        try:
            raw = b"".join(self.vocab[i] for i in ids)
        except KeyError as exc:
            raise ValueError(
                f"id {exc.args[0]!r} is outside this tokenizer's vocabulary "
                f"(size {self.vocab_size})"
            ) from None

        # errors="replace", not "strict": a sampled id sequence can end mid
        # character, and that's a normal thing for a generation to do rather than
        # a reason to raise.
        return raw.decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        # Nothing to record — the 256 bytes are the same everywhere.
        return {"kind": self.kind}

    @classmethod
    def from_dict(cls, data: dict) -> "ByteTokenizer":
        return cls()

    def __eq__(self, other: object) -> bool:
        # Not a dataclass (subclasses carry mutable merge state), so equality is
        # spelled out: two tokenizers are the same if they'd rebuild the same way.
        return type(other) is type(self) and other.to_dict() == self.to_dict()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(vocab_size={self.vocab_size})"
