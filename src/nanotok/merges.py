"""Byte-pair functions shared by both BPE tokenizers.

Keeping them in one place means the trained tokenizer and the tiktoken-based one
use exactly the same merge logic, which is what makes their outputs comparable.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import regex


def text_to_bytes(text: str) -> list[int]:
    """UTF-8 bytes of `text` as ints, the starting point for every BPE vocabulary."""
    tokens = text.encode("utf-8")
    tokens = list(map(int, tokens))
    return tokens


def find_largest(groups: list[list[int]]) -> tuple[int, tuple[int, int]]:
    """Most frequent adjacent pair across all groups, and its count.

    Pairs are counted within each group but never across two groups, so a merge
    can't span a regex split boundary.
    """
    d = defaultdict(int)
    top, top_pair = 0, 0
    # can use zip(st, st[1:])
    # just build dict and do max(d, key=d.get)
    for group in groups:
        for i in range(len(group) - 1):
            pair = (group[i], group[i + 1])
            d[pair] += 1
            if d[pair] > top:
                top, top_pair = d[pair], pair

    return top, top_pair


def merge_groups(groups: list[list[int]], tp: tuple[int, int], new_token: int) -> list[list[int]]:
    for idx, group in enumerate(groups):
        groups[idx] = merge(group, tp, new_token)

    return groups


def merge(st: list[int], tp: tuple[int, int], new_token: int) -> list[int]:
    """Replace every non-overlapping occurrence of `tp` with `new_token`."""
    new_st = []
    idx = 0
    while idx < len(st) - 1:
        if (st[idx], st[idx + 1]) == tp:
            new_st.append(new_token)
            idx += 2
        else:
            new_st.append(st[idx])
            idx += 1

    # the loop stops one short, so a final unpaired token still has to be kept
    if idx == len(st) - 1:
        new_st.append(st[idx])

    return new_st


def build_vocab(
    base_vocab: dict[int, bytes], merge_map: dict[tuple[int, int], int]
) -> dict[int, bytes]:
    """id -> bytes, for the base vocabulary plus every merged token.

    Takes the base vocabulary as an argument rather than assuming the 256 bytes,
    because tokenizers with special tokens start from more than that. Returns a
    new dict rather than modifying `base_vocab`.

    Merges are added in order of their new id, not dict order, since each merge
    refers to two ids that have to exist already. Dict order happens to be id
    order for both current callers, but that's not guaranteed.
    """
    vocab = dict(base_vocab)

    for (p0, p1), idx in sorted(merge_map.items(), key=lambda item: item[1]):
        vocab[idx] = vocab[p0] + vocab[p1]

    return vocab


def validate_specials(specials: Iterable[str]) -> list[str]:
    """Check the special tokens are distinct and non-empty, and raise if not.

    Silently fixing either problem would change the ids: removing a duplicate
    or an empty string shifts every later special down by one. (An empty special
    would also match everywhere.)
    """
    specials = list(specials)

    if any(not s for s in specials):
        raise ValueError(f"special tokens must be non-empty, got {specials!r}")

    duplicates = {s for s in specials if specials.count(s) > 1}
    if duplicates:
        raise ValueError(
            f"special tokens must be distinct; {sorted(duplicates)!r} repeated "
            f"in {specials!r}"
        )

    return specials


def split_on_specials(text: str, specials: Iterable[str]) -> list[str]:
    """Split `text` into ordinary and special pieces, keeping the specials.

    This runs before the regex split, so a special never reaches `apply_merges`.
    That means ordinary text can't produce a special's id, and no merge can
    include one. If it ran after, `<|eot|>` would just be seven characters that
    the merges could put back together.

    Specials are matched longest first, because regex alternation takes the
    first match rather than the longest. Otherwise, with `<|end|>` listed before
    `<|end|>of<|end|>`, the short one would match the start of the long one.
    This ordering doesn't affect the ids, which still follow declaration order.
    """
    specials = list(specials)
    if not specials:
        return [text]

    pattern = regex.compile(
        "(" + "|".join(regex.escape(s) for s in sorted(specials, key=len, reverse=True)) + ")"
    )
    # `split` with a capturing group keeps the matched specials in the result.
    # It also adds empty strings around matches, which we drop.
    return [part for part in pattern.split(text) if part]


def apply_merges(tokens: list[int], merge_map: dict[tuple[int, int], int]) -> list[int]:
    """Encode one chunk by repeatedly merging the lowest-ranked pair present.

    Lowest rank, not leftmost or most frequent: the rank is the order the merges
    were learned in, so applying them in that order reproduces the training-time
    tokenization. Getting this wrong doesn't crash, it just gives plausible but
    wrong ids, which is why the tests compare against tiktoken.
    """
    while True:
        best_rank = float("inf")
        pair = None

        for candidate in zip(tokens, tokens[1:]):
            rank = merge_map.get(candidate, float("inf"))
            if rank < best_rank:
                best_rank, pair = rank, candidate

        if pair is None:
            return tokens

        tokens = merge(tokens, pair, best_rank)
