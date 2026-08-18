"""Tokenizers: text <-> ids, behind one interface.

    CharTokenizer          one id per character, alphabet from a corpus or stated
    ByteTokenizer          one id per byte, no training, no vocabulary to persist
    RegexTokenizer         BPE trained here, over regex-split chunks
    TiktokenBPETokenizer   BPE run here, over a tiktoken encoding's learned ranks

The last two are byte-level and share `ByteTokenizer`'s decode path; the first is
a sibling, not a relative — see `base.py` for why that isn't an oversight.

`from_dict` is the inverse of the `to_dict` every tokenizer implements, and the
reason a run directory can record which tokenizer produced its checkpoints.
"""

import hashlib
import json

from .base import Tokenizer
from .byte import ByteTokenizer
from .char import CharTokenizer
from .regex_bpe import NO_SPLIT_PATTERN, RegexTokenizer
from .tiktoken_bpe import TiktokenBPETokenizer

# Keyed by each class's `kind` tag. Separate from the config-level tokenizer
# table that stage B will add: that one maps a config name to a *builder* that
# may need the corpus, this one rebuilds an already-built tokenizer from what it
# wrote down. Same spirit, different question.
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
    """sha256 of the reconstruction recipe — a tokenizer's identity in one string.

    Goes into a run's provenance, where `vocab_size` alone can't tell two BPE
    vocabularies apart, and will key stage C's token cache. `sort_keys` because
    the fingerprint has to survive a round trip through JSON and a dict whose
    insertion order changed.
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
