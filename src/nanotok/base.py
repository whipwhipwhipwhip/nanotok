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

    #: The tag `to_dict` writes and `nanotok.from_dict` dispatches on. Set by
    #: each concrete subclass; the package-level `from_dict` builds its table
    #: from these, so two tokenizers may not share one.
    kind: ClassVar[str]

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """One past the highest id `encode` can emit, or the padded size.

        This sizes the embedding table and the LM head, so it is a fact about the
        model as much as about the text — which is why it's derived here rather
        than chosen in a config file. One past the highest rather than a count,
        because an id space with a gap in it (cl100k's, once its specials are
        declared) still needs a row for the gap.

        `pad_vocab_to` is the same statement made deliberately: see
        `pad_vocab_size` below for why a tokenizer, rather than the model, is
        where that number belongs.
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


def pad_vocab_size(unpadded_size: int, pad_vocab_to: int | None) -> int:
    """The vocabulary size a tokenizer reports, padded up if it was asked to be.

    Padding buys nothing linguistic: the ids it adds are unreachable, and the
    embedding rows behind them train on nothing. What it buys is a `vocab_size`
    that factorises the way a GPU wants — 50257 is 50257, while 50304 is 64 x 786
    — which is worth a measurable amount of throughput on the matmuls either side
    of the model, and costs a few rows nobody reads.

    That makes a padded id the same kind of object as the hole cl100k leaves at
    100256: inside `vocab_size`, outside `vocab`, decodable to nothing. The only
    difference is that this one is on purpose.

    **Why the tokenizer owns this rather than the model.** `vocab_size` is the
    single number the embedding table is built from, so a model that padded
    privately would disagree with the tokenizer that named its ids, and the
    disagreement would surface as a checkpoint that won't load rather than as an
    error anybody can read. Recording it in the recipe also puts it in the
    fingerprint, which is what stops a padded run and an unpadded one from
    sharing a cache entry.

    Absolute rather than a multiple to round to: the number that matters is the
    one the embedding table ends up being, and stating 50304 says it without the
    reader having to know the unpadded size to work it out. A `pad_to_multiple_of`
    spelling can be layered on later — it would resolve to one of these.
    """
    if pad_vocab_to is None:
        return unpadded_size

    if not isinstance(pad_vocab_to, int) or pad_vocab_to < 1:
        raise ValueError(
            f"pad_vocab_to must be a positive integer, got {pad_vocab_to!r}"
        )

    # Refused rather than quietly returning `max(unpadded, target)`: the caller
    # asked for an embedding table of a stated size, and handing back a different
    # number is how a run ends up the wrong shape with nothing to point at. Equal
    # is fine — it means the vocabulary already landed where you wanted it.
    if pad_vocab_to < unpadded_size:
        raise ValueError(
            f"pad_vocab_to={pad_vocab_to} is below the {unpadded_size} ids this "
            f"tokenizer already has; padding can only grow the id space"
        )

    return pad_vocab_to


def pad_record(pad_vocab_to: int | None) -> dict:
    """The padding entry for `to_dict`, and nothing at all when unpadded.

    Omitted rather than written as null so that an unpadded tokenizer serialises
    exactly as it did before padding existed — its fingerprint is in run
    directories already, and a key that changes every hash to record the absence
    of a feature would invalidate all of them.
    """
    return {} if pad_vocab_to is None else {"pad_vocab_to": pad_vocab_to}


def undecodable_id(token: int, unpadded_size: int, pad_vocab_to: int | None) -> ValueError:
    """The error `decode` raises for an id it holds no bytes for.

    Padding makes the plain "outside the vocabulary" message a lie for exactly
    the ids padding added: they are *inside* `vocab_size` — that is the whole
    point of them — and still decode to nothing. Saying so names the actual bug,
    which is a sampler drawing from logits that were never masked.
    """
    if (
        pad_vocab_to is not None
        and isinstance(token, int)
        and unpadded_size <= token < pad_vocab_to
    ):
        return ValueError(
            f"id {token!r} is padding: it exists only to round this tokenizer's "
            f"vocabulary up to {pad_vocab_to} and has no text of its own. Mask "
            f"ids at or above {unpadded_size} out of the model's logits."
        )

    return ValueError(
        f"id {token!r} is outside this tokenizer's vocabulary "
        f"(size {pad_vocab_size(unpadded_size, pad_vocab_to)})"
    )
