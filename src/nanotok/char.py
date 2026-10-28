"""Character-level tokenizer: one id per distinct character.

There are two ways to build one:

`from_text` gets the alphabet from the corpus, as in the nanoGPT lecture. The ids
then depend on the corpus, so if it gains a new character, every id after it
shifts.

`from_alphabet` takes the alphabet as an argument. For a task with a known, fixed
set of characters (e.g. "0123456789+=\\n" for addition) this is the better
choice: the ids are the same on every run, so old checkpoints still work after
the training data changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Iterable, Sequence

from .base import Tokenizer, pad_record, pad_vocab_size, undecodable_id


@dataclass(frozen=True)
class CharTokenizer(Tokenizer):
    """Frozen, since changing the vocabulary after a model's embedding table
    has been sized from it would break the model."""

    stoi: dict[str, int]
    itos: dict[int, str]
    #: Report this as `vocab_size` instead; see `base.pad_vocab_size`. It's a
    #: dataclass field so that it's included in `__eq__`: two tokenizers with
    #: the same alphabet but different padding need different embedding tables.
    pad_vocab_to: int | None = None

    kind: ClassVar[str] = "char"

    # ---- construction -----------------------------------------------------

    @classmethod
    def _from_chars(
        cls, chars: Iterable[str], pad_vocab_to: int | None = None
    ) -> "CharTokenizer":
        """Assigns the ids. The characters are sorted so the same set always
        gets the same ids, whatever order they came in."""
        chars = sorted(set(chars))

        if not chars:
            raise ValueError("cannot build a CharTokenizer over an empty alphabet")

        return cls(
            stoi={ch: i for i, ch in enumerate(chars)},
            itos={i: ch for i, ch in enumerate(chars)},
            pad_vocab_to=pad_vocab_to,
        )

    @classmethod
    def from_text(cls, text: str, pad_vocab_to: int | None = None) -> "CharTokenizer":
        """Alphabet taken from the corpus."""
        return cls._from_chars(text, pad_vocab_to)

    @classmethod
    def from_alphabet(
        cls, alphabet: str, pad_vocab_to: int | None = None
    ) -> "CharTokenizer":
        """Alphabet given directly. Duplicates and order don't matter."""
        return cls._from_chars(alphabet, pad_vocab_to)

    # ---- the interface ----------------------------------------------------

    @property
    def unpadded_vocab_size(self) -> int:
        """The number of characters, not counting padding."""
        return len(self.itos)

    @property
    def vocab_size(self) -> int:
        return pad_vocab_size(self.unpadded_vocab_size, self.pad_vocab_to)

    @property
    def alphabet(self) -> str:
        """The characters in id order, as written by `to_dict`.

        Stops at `unpadded_vocab_size`, since padding ids have no character.
        """
        return "".join(self.itos[i] for i in range(self.unpadded_vocab_size))

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:
            # A plain KeyError only names the character. Usually the cause is a
            # fixed alphabet that's missing a character the data contains.
            raise ValueError(
                f"character {exc.args[0]!r} is not in this tokenizer's alphabet "
                f"({self.alphabet!r}); add it to the alphabet or fix the input"
            ) from None

    def decode(self, ids: Sequence[int]) -> str:
        try:
            return "".join(self.itos[i] for i in ids)
        except KeyError as exc:
            raise undecodable_id(
                exc.args[0], self.unpadded_vocab_size, self.pad_vocab_to
            ) from None

    # ---- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "alphabet": self.alphabet,
            **pad_record(self.pad_vocab_to),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        # Not `from_alphabet`, because that sorts and this should reproduce the
        # ids exactly as saved. The two give the same result at the moment
        # (ids are assigned in sorted order), but this doesn't rely on that.
        chars = data["alphabet"]
        return cls(
            stoi={ch: i for i, ch in enumerate(chars)},
            itos={i: ch for i, ch in enumerate(chars)},
            pad_vocab_to=data.get("pad_vocab_to"),
        )
