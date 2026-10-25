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

**Vocabulary padding lives here too**, and for the third time for the same
reason: rounding `vocab_size` up to a GPU-friendly number is arithmetic on one
integer, identical whether the ids below it were merged here or came from
tiktoken. `base.pad_vocab_size` says why a tokenizer owns that number at all.
"""

from __future__ import annotations

from typing import ClassVar, Sequence

from .base import Tokenizer, pad_record, pad_vocab_size, undecodable_id
from .merges import split_on_specials, text_to_bytes


class ByteTokenizer(Tokenizer):
    """Byte-level identity. Subclasses grow `vocab` past 256 and override `_encode_ordinary`."""

    kind: ClassVar[str] = "byte"

    def __init__(self, pad_vocab_to: int | None = None) -> None:
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        #: text -> id, for ids that are declared rather than merged into
        #: existence. Empty here; subclasses that accept specials fill it.
        self.special_map: dict[str, int] = {}
        #: Report this `vocab_size` instead, if it's larger. Config, not state:
        #: it survives `train`, which resets everything else here.
        self.pad_vocab_to = pad_vocab_to

    @property
    def unpadded_vocab_size(self) -> int:
        # Derived from the table rather than stored, so it's right before, during
        # and after training. The old RegexTokenizer set `self.vocab_size` inside
        # `train`, which left an untrained instance with no attribute at all.
        #
        # Highest id plus one, not the number of entries: this sizes an embedding
        # table, so a gap in the id space still has to be paid for. cl100k's
        # specials start at 100257 and leave 100256 unused, so counting entries
        # would hand the model a table one row short of the ids it must embed.
        return max(self.vocab) + 1

    @property
    def vocab_size(self) -> int:
        # The number that sizes the embedding table, padding included. Kept
        # separate from `unpadded_vocab_size` above, which is what the merge
        # machinery counts with: `train` numbers new merges from the ids that
        # actually exist, and padding must not move them.
        return pad_vocab_size(self.unpadded_vocab_size, self.pad_vocab_to)

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
            raise undecodable_id(
                exc.args[0], self.unpadded_vocab_size, self.pad_vocab_to
            ) from None

        # errors="replace", not "strict": a sampled id sequence can end mid
        # character, and that's a normal thing for a generation to do rather than
        # a reason to raise.
        return raw.decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        # Nothing to record but the padding — the 256 bytes are the same
        # everywhere, and the padding is the only thing that can make two byte
        # tokenizers differently shaped.
        return {"kind": self.kind, **pad_record(self.pad_vocab_to)}

    @classmethod
    def from_dict(cls, data: dict) -> "ByteTokenizer":
        return cls(pad_vocab_to=data.get("pad_vocab_to"))

    def __eq__(self, other: object) -> bool:
        # Not a dataclass (subclasses carry mutable merge state), so equality is
        # spelled out: two tokenizers are the same if they'd rebuild the same way.
        return type(other) is type(self) and other.to_dict() == self.to_dict()

    def __repr__(self) -> str:
        # Padding shown as where it came from, not just where it landed: a bare
        # 50304 reads as a vocabulary, and it isn't one.
        padded = (
            "" if self.pad_vocab_to is None else f" padded from {self.unpadded_vocab_size}"
        )
        return f"{type(self).__name__}(vocab_size={self.vocab_size}{padded})"
