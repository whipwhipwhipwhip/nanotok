"""Tokenizers that convert text to ids and back, all with the same interface.

    CharTokenizer          one id per character, alphabet from a corpus or given
    ByteTokenizer          one id per byte, nothing to train
    RegexTokenizer         BPE trained here, over regex-split chunks
    TiktokenBPETokenizer   our BPE, using a tiktoken encoding's vocabulary

The two BPE tokenizers subclass `ByteTokenizer`. `CharTokenizer` doesn't (see
`base.py` for why).

`from_dict` rebuilds any tokenizer from its `to_dict()` output, so a training run
can save which tokenizer it used.
"""

import hashlib
import json

from .base import Tokenizer
from .byte import ByteTokenizer
from .char import CharTokenizer
from .regex_bpe import NO_SPLIT_PATTERN, RegexTokenizer
from .tiktoken_bpe import TiktokenBPETokenizer

# Each class's `kind` tag -> the class, for `from_dict`.
_KINDS: dict[str, type[Tokenizer]] = {
    cls.kind: cls
    for cls in (CharTokenizer, ByteTokenizer, RegexTokenizer, TiktokenBPETokenizer)
}


def from_dict(data: dict) -> Tokenizer:
    """Rebuild whichever tokenizer wrote `data`."""
    try:
        kind = data["kind"]
    except KeyError:
        raise ValueError(
            f"tokenizer record has no 'kind' key; got {sorted(data)}"
        ) from None

    try:
        cls = _KINDS[kind]
    except KeyError:
        raise ValueError(
            f"unknown tokenizer kind {kind!r}; expected one of {sorted(_KINDS)}"
        ) from None

    return cls.from_dict(data)


def fingerprint(tokenizer: Tokenizer) -> str:
    """sha256 of `tokenizer.to_dict()`, to identify a tokenizer in one string.

    Useful for recording which tokenizer a run used (`vocab_size` alone can't
    tell two BPE vocabularies apart) or as a cache key. Keys are sorted so the
    hash doesn't depend on dict order.
    """
    recipe = json.dumps(tokenizer.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(recipe).hexdigest()


__all__ = [
    "NO_SPLIT_PATTERN",
    "ByteTokenizer",
    "CharTokenizer",
    "RegexTokenizer",
    "TiktokenBPETokenizer",
    "Tokenizer",
    "fingerprint",
    "from_dict",
]
