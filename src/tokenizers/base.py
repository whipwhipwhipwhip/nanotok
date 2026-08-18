"""The tokenizer contract: text <-> ids, plus a way to write itself down.

One interface over two different ID spaces. `CharTokenizer` numbers the distinct
characters of a corpus; everything BPE numbers the 256 bytes and then merges
upward. Those are not refinements of each other — a byte tokenizer is not a
character tokenizer with more entries — so they meet here, at the interface, and
nowhere else. In particular `CharTokenizer` is deliberately *not* the base class:
a `RegexTokenizer` inheriting from it would inherit a vocabulary it has to throw
away before it can do anything.

What is deliberately absent:

`train`. Two of the four implementations have nothing to train — a character
vocabulary is read off the corpus in a single pass, and a tiktoken-backed one
arrives already trained. Hoisting `train` here would buy a shared signature at
the cost of two subclasses raising NotImplementedError to satisfy a contract
nobody calls through. Trainability is the builder's problem, not the interface's.

Special tokens. They exist, but they are a property of the byte-level branch
rather than of the interface: both BPE tokenizers accept them, `CharTokenizer`
does not, and nothing here needs to know which. They are declared in the config
and never learned from a corpus, which is what keeps their ids a fact about the
recipe — see `byte.py` for the mechanism and `tiktoken_bpe.py` for why the two
implementations disagree about where the ids come from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Sequence


class Tokenizer(ABC):
    """What every tokenizer promises."""

    #: The tag `to_dict` writes and `tokenizers.from_dict` dispatches on. Set by
    #: each concrete subclass; the package-level `from_dict` builds its table
    #: from these, so two tokenizers may not share one.
    kind: ClassVar[str]

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """One past the highest id `encode` can emit.

        This sizes the embedding table and the LM head, so it is a fact about the
        model as much as about the text — which is why it's derived here rather
        than chosen in a config file. One past the highest rather than a count,
        because an id space with a gap in it (cl100k's, once its specials are
        declared) still needs a row for the gap.
        """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Text -> ids. Raises rather than substituting a placeholder for input
        the vocabulary can't represent: silently mapping unknown input to some
        fallback id trains a model on a lie."""

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Ids -> text. Not a strict inverse of `encode` for byte-level
        tokenizers: an id sequence that splits a multi-byte character produces
        U+FFFD rather than an error, because partial output is what sampling
        mid-token legitimately looks like."""

    @abstractmethod
    def to_dict(self) -> dict:
        """A recipe for rebuilding this tokenizer, not a dump of its state.

        `TiktokenBPETokenizer` records an encoding name rather than 50k merges;
        `CharTokenizer` records its alphabet rather than two dicts. Two rules
        this has to keep, because a run record depends on them:

        * JSON-serialisable, since it goes to `runs/<id>/tokenizer.json` — that
          file is the only thing standing between a checkpoint and an
          undecodable pile of integers, given `*.txt` is gitignored.
        * Deterministic, since its sha256 is the fingerprint that identifies a
          tokenizer in provenance and keys the token cache.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "Tokenizer":
        """Rebuild from `to_dict`. `from_dict(t.to_dict())` must equal `t`."""
