"""BPE that runs our merge loop, seeded from a tiktoken encoding's learned ranks.

One class for every tiktoken encoding — `gpt2` and `cl100k_base` differ only in
the ranks and the split pattern, both of which come out of the encoding itself.

**Why this rather than calling tiktoken?** Because building it is the point; this
is a learning repo. `tiktoken.get_encoding(name).encode` is a drop-in replacement
behind the `Tokenizer` ABC — a second entry in the tokenizer table, no other
change anywhere — and is roughly a hundred times faster, since it does the merge
loop in Rust. Swap to it if encode cost ever dominates a real training run; the
tiktoken-equivalence tests in `tests/nanotok/test_bpe.py` are what make that a
swap rather than a rewrite, so they have to keep passing.

**Why the merges have to be recovered.** `mergeable_ranks` maps a finished byte
string to its rank (`b"ing" -> 278`), which has already thrown away *which two
pieces* were joined to make it — and the pair is exactly what a merge loop needs.
`recover_merges` re-derives it by re-running BPE on each token restricted to
lower ranks. Calling tiktoken directly needs none of this; running your own loop
needs all of it.

**Why the byte shuffle.** Neither GPT-2 nor GPT-4 numbers the 256 single-byte
tokens in byte order — under `gpt2`, byte `0x00` is rank 188. So raw bytes are
permuted into rank space before merging, and permuted back after decoding.

**Special tokens are opt-in, and keep the encoding's own ids.** `<|endoftext|>`
and friends live outside `mergeable_ranks`, in a table of their own, so a
tokenizer that declares none reports 50256 for GPT-2 rather than tiktoken's
50257 — which is the right answer for a run that never emits one, and the wrong
size of embedding table for a run that does. Declaring them by name takes the
ids OpenAI assigned (`<|endoftext|>` is 50256 under gpt2, 100257 under cl100k)
rather than numbering them above the ranks, because those ids are what makes our
output interchangeable with `tiktoken.encode`. Under cl100k that leaves 100256
unused: a hole in the id space, which `vocab_size` counts, since an embedding
table is sized by the highest id and not by how many are reachable.

**Padding is the same hole, dug on purpose.** `pad_vocab_to=50304` takes GPT-2's
50256 (or 50257, with `<|endoftext|>` declared) up to a number with friendlier
factors, which is worth real throughput on the matmuls either side of the model
and costs a handful of rows nothing ever reads. It changes `vocab_size` and
nothing else — the ids this class emits stay the ids `tiktoken.encode` emits,
which is the property the whole module exists to keep. See `base.pad_vocab_size`.
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
        # The encoding's own pattern by default, so the split can't drift out of
        # step with the ranks — GPT-2's pattern is case-sensitive and GPT-4's
        # isn't, and using one with the other's merges produces ids that look
        # fine and are wrong.
        self.splitter = pat_str if split_pattern is None else split_pattern

        # Copies: `_load_encoding` is cached, and one instance mutating these
        # would silently reshape every other instance of the same encoding.
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
        """Take the encoding's own ids for the specials this run declared.

        Not numbered above the ranks the way `RegexTokenizer` numbers its own:
        these ids already exist, and inventing different ones would break the
        equivalence with `tiktoken.encode` that the whole class is here to keep.

        An unknown name is an error rather than a new token, for the same
        reason — a tokenizer cannot mint an id in a vocabulary it didn't train.
        Use `RegexTokenizer` if you want to declare your own.
        """
        unknown = [s for s in self.special_tokens if s not in available]
        if unknown:
            raise ValueError(
                f"{self.encoding_name} defines no special token {unknown[0]!r}; "
                f"expected one of {sorted(available)}"
            )

        self.special_map = {s: available[s] for s in self.special_tokens}

        # Stored permuted, like every other entry in this table: `decode`
        # un-permutes the whole concatenated byte string in one pass at the end,
        # so a special written in raw bytes would come back out scrambled.
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

        # Un-permute after concatenating, not before: the permutation is
        # per-byte, so it commutes with joining, and doing it once at the end is
        # one pass instead of one per token.
        raw = bytes(self.inverse_byte_shuffle[b] for b in shuffled)
        return raw.decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        # The encoding name, not 50k merges: tiktoken is a dependency, the ranks
        # are fixed, and a 2MB tokenizer.json in every run directory would be
        # a copy of something that can't drift.
        return {
            "kind": self.kind,
            "encoding": self.encoding_name,
            "splitter": self.splitter,
            # Which specials this run declared, not which the encoding defines:
            # the difference is `vocab_size`, so it's part of the recipe.
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
    """Everything we take from tiktoken, recovered once per encoding.

    Cached because `recover_merges` re-runs BPE on all 50k+ tokens — seconds, not
    milliseconds, and it produces the same answer every time.

    `_mergeable_ranks` and `_pat_str` are private tiktoken attributes. That's the
    price of reimplementing the algorithm: the public API only offers `encode`,
    which is the thing we're deliberately not calling.
    """
    enc = tiktoken.get_encoding(encoding_name)
    ranks = enc._mergeable_ranks

    merge_map = recover_merges(ranks)
    byte_shuffle = {i: ranks[bytes([i])] for i in range(256)}

    # `_special_tokens` is the table `mergeable_ranks` deliberately isn't part
    # of: names to ids that no merge can ever produce. Public in the sense that
    # `enc.encode(..., allowed_special=...)` names them, private as an attribute.
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
