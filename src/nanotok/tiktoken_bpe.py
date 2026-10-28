"""Our BPE merge loop, using the learned ranks from a tiktoken encoding.

One class covers every tiktoken encoding. `gpt2` and `cl100k_base` only differ
in their ranks and split pattern, and both come from the encoding itself.

**Why not just call tiktoken?** Because this is a learning repo, and the point
is to write it myself. `tiktoken.get_encoding(name).encode` could be wrapped in
the same `Tokenizer` interface with no other changes, and would be roughly a
hundred times faster since its merge loop is in Rust. If encoding speed ever
matters for a training run, swap to that. The tiktoken comparison tests in
`tests/nanotok/test_bpe.py` are what guarantee the two give the same ids.

**Why the merges have to be recovered.** `mergeable_ranks` maps each finished
byte string to its rank (`b"ing" -> 278`), but doesn't say which two pieces were
joined to make it, and the pair is what a merge loop needs. `recover_merges`
works it out by re-running BPE on each token using only the lower ranks.

**Why the bytes are shuffled.** Neither GPT-2 nor GPT-4 gives the 256 single
bytes ranks in byte order (under `gpt2`, byte `0x00` is rank 188). So raw bytes
are mapped into rank space before merging, and mapped back after decoding.

**Special tokens are opt-in, and keep the encoding's own ids.** `<|endoftext|>`
and the other specials aren't in `mergeable_ranks`; tiktoken keeps them in a
separate table. So a tokenizer with no specials declared has a vocab size of
50256 for GPT-2, not tiktoken's 50257. That's correct if the run never uses
`<|endoftext|>`, and one too small if it does. Declaring a special by name uses
the id OpenAI assigned (`<|endoftext|>` is 50256 under gpt2, 100257 under
cl100k) rather than numbering it after the ranks, so the output still matches
`tiktoken.encode`. Under cl100k this leaves 100256 unused, and `vocab_size`
still counts it, since an embedding table is sized by the highest id.

**Padding.** `pad_vocab_to=50304` rounds GPT-2's 50256 (or 50257 with
`<|endoftext|>`) up to a size that's faster on a GPU. It changes `vocab_size`
and nothing else, so the ids still match `tiktoken.encode`. See
`base.pad_vocab_size`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import ClassVar, Sequence

import regex
import tiktoken

from .base import pad_record, undecodable_id
from .byte import ByteTokenizer
from .merges import (
    apply_merges,
    build_vocab,
    text_to_bytes,
    validate_specials,
)


class TiktokenBPETokenizer(ByteTokenizer):
    """Our BPE, OpenAI's vocabulary."""

    kind: ClassVar[str] = "tiktoken_bpe"

    def __init__(
        self,
        encoding_name: str,
        split_pattern: str | None = None,
        special_tokens: list[str] | None = None,
        pad_vocab_to: int | None = None,
    ) -> None:
        super().__init__(pad_vocab_to=pad_vocab_to)

        merge_map, byte_shuffle, pat_str, available_specials = _load_encoding(encoding_name)

        self.encoding_name = encoding_name
        # Use the encoding's own pattern by default, so the split matches the
        # ranks. GPT-2's pattern is case-sensitive and GPT-4's isn't, and using
        # one with the other's merges gives ids that look fine but are wrong.
        self.splitter = pat_str if split_pattern is None else split_pattern

        # Copies, because `_load_encoding` is cached and every instance of the
        # same encoding shares its return values.
        self.merge_map = dict(merge_map)
        self.byte_shuffle = dict(byte_shuffle)
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}
        self.vocab = build_vocab(self.vocab, self.merge_map)

        self.special_tokens = validate_specials(special_tokens or [])
        self._register_specials(available_specials)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        raise NotImplementedError(
            "this tokenizer's merges come from a trained tiktoken encoding; "
            "use RegexTokenizer to train your own"
        )

    def _register_specials(self, available: dict[str, int]) -> None:
        """Look up the encoding's ids for the declared special tokens.

        Unlike `RegexTokenizer`, these aren't numbered after the merges: the
        ids already exist, and using different ones would stop the output
        matching `tiktoken.encode`.

        For the same reason, a name the encoding doesn't define is an error
        rather than a new token. Use `RegexTokenizer` to define your own.
        """
        unknown = [s for s in self.special_tokens if s not in available]
        if unknown:
            raise ValueError(
                f"{self.encoding_name} defines no special token {unknown[0]!r}; "
                f"expected one of {sorted(available)}"
            )

        self.special_map = {s: available[s] for s in self.special_tokens}

        # Stored in rank space like everything else in `vocab`, because `decode`
        # un-shuffles all the bytes at the end. Raw bytes would come out
        # scrambled.
        for special, token in self.special_map.items():
            self.vocab[token] = bytes(self._to_rank_space(special))

    def _encode_ordinary(self, text: str) -> list[int]:
        return [
            token
            for match in regex.finditer(self.splitter, text)
            for token in apply_merges(self._to_rank_space(match.group()), self.merge_map)
        ]

    def _to_rank_space(self, chunk: str) -> list[int]:
        return [self.byte_shuffle[b] for b in text_to_bytes(chunk)]

    def decode(self, ids: Sequence[int]) -> str:
        try:
            shuffled = b"".join(self.vocab[i] for i in ids)
        except KeyError as exc:
            raise undecodable_id(
                exc.args[0], self.unpadded_vocab_size, self.pad_vocab_to
            ) from None

        # Un-shuffle once after joining rather than per token. The shuffle is
        # per byte, so the result is the same.
        raw = bytes(self.inverse_byte_shuffle[b] for b in shuffled)
        return raw.decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        # Just the encoding name, not 50k merges. tiktoken is a dependency and
        # its ranks don't change, so there's no need to save a copy.
        return {
            "kind": self.kind,
            "encoding": self.encoding_name,
            "splitter": self.splitter,
            # The specials this tokenizer declared, not all the encoding
            # defines. They change `vocab_size`, so they're part of the recipe.
            "special_tokens": list(self.special_tokens),
            **pad_record(self.pad_vocab_to),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TiktokenBPETokenizer":
        return cls(
            data["encoding"],
            data["splitter"],
            data["special_tokens"],
            data.get("pad_vocab_to"),
        )


@lru_cache(maxsize=None)
def _load_encoding(encoding_name: str):
    """Everything we need from tiktoken, loaded once per encoding.

    Cached because `recover_merges` re-runs BPE on all 50k+ tokens, which takes
    a few seconds and always gives the same answer.

    `_mergeable_ranks`, `_pat_str` and `_special_tokens` are private tiktoken
    attributes. The public API only offers `encode`, which is the thing we're
    reimplementing.
    """
    enc = tiktoken.get_encoding(encoding_name)
    ranks = enc._mergeable_ranks

    merge_map = recover_merges(ranks)
    byte_shuffle = {i: ranks[bytes([i])] for i in range(256)}

    return merge_map, byte_shuffle, enc._pat_str, dict(enc._special_tokens)


def bpe(mergeable_ranks, token, max_rank):
    # helper function used in recover_merges() to reconstruct the merge forest
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts


def recover_merges(mergeable_ranks):
    # the `merges` are already the byte sequences in their merged state.
    # so we have to recover the original pairings. We can do this by doing
    # a small BPE training run on all the tokens, in their order.
    # also see https://github.com/openai/tiktoken/issues/60
    # also see https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue  # skip raw bytes
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        # recover the integer ranks of the pair
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank

    return merges
