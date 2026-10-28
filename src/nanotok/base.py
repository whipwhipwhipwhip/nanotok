"""The interface every tokenizer implements: encode, decode, and save/load.

`CharTokenizer` numbers the distinct characters of a corpus, while the BPE
tokenizers start from the 256 bytes and merge upward. Neither id space is an
extension of the other, so they only share this interface. In particular,
`CharTokenizer` isn't the base class: a `RegexTokenizer` that inherited from it
would get a character vocabulary it has no use for.

Two things are left out of the interface on purpose:

`train`. Only `RegexTokenizer` actually trains. A character vocabulary is just
read off the corpus, and a tiktoken-backed tokenizer comes already trained, so
putting `train` here would mean two subclasses raising NotImplementedError.

Special tokens. Both BPE tokenizers support them, but `CharTokenizer` doesn't, so
they're handled in `byte.py` instead. They're always declared up front, never
learned from a corpus, so their ids only depend on the config. See
`tiktoken_bpe.py` for why the two BPE tokenizers assign those ids differently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Sequence


class Tokenizer(ABC):
    """Base class for all tokenizers."""

    #: The tag `to_dict` writes and `nanotok.from_dict` uses to pick the class.
    #: Each concrete subclass sets its own, and they must all be different.
    kind: ClassVar[str]

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """One more than the highest id `encode` can emit, or the padded size.

        This is the size of the embedding table and the LM head. It's the
        highest id plus one rather than the number of tokens because the id
        space can have gaps (cl100k does, once its specials are declared), and
        the embedding table still needs a row for each gap.

        If `pad_vocab_to` is set, this is that number instead; see
        `pad_vocab_size` below.
        """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Text -> ids. Raises on input the vocabulary can't represent, instead
        of mapping it to some fallback id, so bad data doesn't end up in
        training unnoticed."""

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Ids -> text. For byte-level tokenizers this isn't a strict inverse
        of `encode`: ids that split a multi-byte character decode to U+FFFD
        instead of raising, since a model can sample sequences like that."""

    @abstractmethod
    def to_dict(self) -> dict:
        """A recipe for rebuilding this tokenizer, not a dump of its state.

        For example, `TiktokenBPETokenizer` stores an encoding name rather than
        50k merges, and `CharTokenizer` stores its alphabet rather than two
        dicts. It has to be:

        * JSON-serialisable, so it can be saved next to a checkpoint (without
          it, the checkpoint's ids can't be decoded).
        * Deterministic, since `nanotok.fingerprint` hashes it.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "Tokenizer":
        """Rebuild from `to_dict`. `from_dict(t.to_dict())` must equal `t`."""


def pad_vocab_size(unpadded_size: int, pad_vocab_to: int | None) -> int:
    """The vocabulary size a tokenizer reports, after padding if requested.

    The padding ids are never produced by `encode`, so their embedding rows
    never get trained. The point is to make `vocab_size` a GPU-friendly number:
    50257 is odd, while 50304 = 64 x 786, which makes the matmuls at
    either end of the model measurably faster for the cost of a few unused rows.

    So a padding id is like the gap cl100k leaves at 100256: inside
    `vocab_size`, but not in `vocab`, and it can't be decoded.

    The tokenizer owns this rather than the model because `vocab_size` is what
    the embedding table is built from. If the model padded on its own, the two
    would disagree, which shows up as a checkpoint that won't load. Keeping it in
    the recipe also means it's part of the fingerprint, so padded and unpadded
    tokenizers don't look the same.

    It's an absolute size rather than a multiple to round to, so the final
    embedding table size is written down directly. A `pad_to_multiple_of`
    option could be added later on top of this.
    """
    if pad_vocab_to is None:
        return unpadded_size

    if not isinstance(pad_vocab_to, int) or pad_vocab_to < 1:
        raise ValueError(
            f"pad_vocab_to must be a positive integer, got {pad_vocab_to!r}"
        )

    # Raise rather than returning `max(unpadded, target)`: the caller asked for
    # a specific embedding table size, and returning something else would give
    # a model of the wrong shape. Equal is fine.
    if pad_vocab_to < unpadded_size:
        raise ValueError(
            f"pad_vocab_to={pad_vocab_to} is below the {unpadded_size} ids this "
            f"tokenizer already has; padding can only grow the id space"
        )

    return pad_vocab_to


def pad_record(pad_vocab_to: int | None) -> dict:
    """The padding entry for `to_dict`, or nothing at all when unpadded.

    Left out rather than written as null so that unpadded tokenizers serialise
    exactly as they did before padding was added, and their old fingerprints
    still match.
    """
    return {} if pad_vocab_to is None else {"pad_vocab_to": pad_vocab_to}


def undecodable_id(token: int, unpadded_size: int, pad_vocab_to: int | None) -> ValueError:
    """The error `decode` raises for an id it has no bytes for.

    Padding ids get their own message, since they *are* inside `vocab_size` and
    "outside the vocabulary" would be misleading. Decoding one usually means the
    sampler didn't mask the padding logits.
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
