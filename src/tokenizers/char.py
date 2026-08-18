"""Character-level tokenizer: one id per distinct character.

Two ways in, and the difference matters more than it looks:

`from_text` reads the alphabet off the corpus, which is what the nanoGPT lecture
does and what `load_data` has always done. The ids it produces are a function of
the corpus, so a corpus that gains a stray character shifts every id above it.

`from_alphabet` states the alphabet up front. For a task with a known, closed
character set — "0123456789+=\\n" for the adder — that's the one you want: the ids
mean the same thing on every run, so a checkpoint stays readable even after the
generator that produced its training data has been edited.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Iterable, Sequence

from .base import Tokenizer


@dataclass(frozen=True)
class CharTokenizer(Tokenizer):
    """Frozen: a tokenizer that changes shape mid-run invalidates the embedding
    table that was sized from it."""

    stoi: dict[str, int]
    itos: dict[int, str]

    kind: ClassVar[str] = "char"

    # ---- construction -----------------------------------------------------

    @classmethod
    def _from_chars(cls, chars: Iterable[str]) -> "CharTokenizer":
        """The one place ids get assigned. Sorted, so that the same set of
        characters always numbers the same way regardless of the order they
        arrived in — otherwise two runs over the same alphabet could disagree
        about what id 7 means."""
        chars = sorted(set(chars))

        if not chars:
            raise ValueError("cannot build a CharTokenizer over an empty alphabet")

        return cls(
            stoi={ch: i for i, ch in enumerate(chars)},
            itos={i: ch for i, ch in enumerate(chars)},
        )

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Alphabet derived from the corpus."""
        return cls._from_chars(text)

    @classmethod
    def from_alphabet(cls, alphabet: str) -> "CharTokenizer":
        """Alphabet stated up front. Duplicates are ignored, order is irrelevant."""
        return cls._from_chars(alphabet)

    # ---- the contract -----------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def alphabet(self) -> str:
        """The vocabulary in id order — what `to_dict` writes down."""
        return "".join(self.itos[i] for i in range(self.vocab_size))

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:
            # The default KeyError says only which character; this says what to
            # do about it, because the usual cause is a fixed alphabet that
            # forgot a character the data generator emits.
            raise ValueError(
                f"character {exc.args[0]!r} is not in this tokenizer's alphabet "
                f"({self.alphabet!r}); add it to the alphabet or fix the input"
            ) from None

    def decode(self, ids: Sequence[int]) -> str:
        try:
            return "".join(self.itos[i] for i in ids)
        except KeyError as exc:
            raise ValueError(
                f"id {exc.args[0]!r} is outside this tokenizer's vocabulary "
                f"(size {self.vocab_size})"
            ) from None

    # ---- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return {"kind": self.kind, "alphabet": self.alphabet}

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        # Deliberately not `from_alphabet`: that sorts, and this must reproduce
        # the ids exactly as written. They agree today, since `alphabet` is
        # emitted in id order and ids were assigned in sorted order — but this
        # one stays correct if that ever stops being true.
        chars = data["alphabet"]
        return cls(
            stoi={ch: i for i, ch in enumerate(chars)},
            itos={i: ch for i, ch in enumerate(chars)},
        )
