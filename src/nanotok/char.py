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

from .base import Tokenizer, pad_record, pad_vocab_size, undecodable_id


@dataclass(frozen=True)
class CharTokenizer(Tokenizer):
    """Frozen: a tokenizer that changes shape mid-run invalidates the embedding
    table that was sized from it."""

    stoi: dict[str, int]
    itos: dict[int, str]
    #: Report this `vocab_size` instead, if it's larger — see `base.pad_vocab_size`.
    #: A field rather than an inherited attribute because the frozen dataclass has
    #: to know about it: it belongs in `__eq__`, since two tokenizers over the same
    #: alphabet with different padding size different embedding tables.
    pad_vocab_to: int | None = None

    kind: ClassVar[str] = "char"

    # ---- construction -----------------------------------------------------

    @classmethod
    def _from_chars(
        cls, chars: Iterable[str], pad_vocab_to: int | None = None
    ) -> "CharTokenizer":
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
            pad_vocab_to=pad_vocab_to,
        )

    @classmethod
    def from_text(cls, text: str, pad_vocab_to: int | None = None) -> "CharTokenizer":
        """Alphabet derived from the corpus."""
        return cls._from_chars(text, pad_vocab_to)

    @classmethod
    def from_alphabet(
        cls, alphabet: str, pad_vocab_to: int | None = None
    ) -> "CharTokenizer":
        """Alphabet stated up front. Duplicates are ignored, order is irrelevant."""
        return cls._from_chars(alphabet, pad_vocab_to)

    # ---- the contract -----------------------------------------------------

    @property
    def unpadded_vocab_size(self) -> int:
        """The characters, and nothing else. What `alphabet` iterates."""
        return len(self.itos)

    @property
    def vocab_size(self) -> int:
        return pad_vocab_size(self.unpadded_vocab_size, self.pad_vocab_to)

    @property
    def alphabet(self) -> str:
        """The vocabulary in id order — what `to_dict` writes down.

        Over `unpadded_vocab_size`, not `vocab_size`: the padded ids stand for no
        character, so walking to `vocab_size` here would raise on a padded
        tokenizer — and a padded alphabet round-tripping through `to_dict` would
        be a different tokenizer anyway.
        """
        return "".join(self.itos[i] for i in range(self.unpadded_vocab_size))

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
        # Deliberately not `from_alphabet`: that sorts, and this must reproduce
        # the ids exactly as written. They agree today, since `alphabet` is
        # emitted in id order and ids were assigned in sorted order — but this
        # one stays correct if that ever stops being true.
        chars = data["alphabet"]
        return cls(
            stoi={ch: i for i, ch in enumerate(chars)},
            itos={i: ch for i, ch in enumerate(chars)},
            pad_vocab_to=data.get("pad_vocab_to"),
        )
