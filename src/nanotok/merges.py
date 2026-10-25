"""Byte-pair primitives, shared by every BPE tokenizer in the package.

These were duplicated verbatim in `bpe/regex_tokenizer.py` and
`bpe/gpt4_tokenizer.py`. One copy, because a fix to `merge` should not have to be
made twice — and because the trained tokenizer and the tiktoken-seeded one
agreeing on what a merge *is* is the whole reason their outputs are comparable.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import regex


def text_to_bytes(text: str) -> list[int]:
    """The bottom of the ID space: every BPE vocabulary starts as the 256 bytes."""
    tokens = text.encode("utf-8")
    tokens = list(map(int, tokens))
    return tokens


def find_largest(groups: list[list[int]]) -> tuple[int, tuple[int, int]]:
    """Most frequent adjacent pair across all groups, and its count.

    Counted across groups but never *between* them: pairs are only formed within
    a group, which is what stops a merge from spanning the regex split boundary.
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
    """id -> bytes, for the starting vocabulary plus everything the merges built.

    Takes the base rather than assuming the 256 bytes, because a tokenizer that
    reserves ids for special tokens starts from more than the bytes and the
    merges are numbered above whatever it started with. Copies rather than
    extending in place: the caller's table is usually the one it's about to
    replace, and aliasing the two makes a half-built vocabulary observable.

    Sorted by the new id rather than trusting the dict's insertion order: a merge
    is defined in terms of two ids that must already exist, so building out of
    rank order raises KeyError on a forward reference. Insertion order happens to
    be rank order for both of today's callers, which is exactly the kind of
    coincidence that breaks silently later.
    """
    vocab = dict(base_vocab)

    for (p0, p1), idx in sorted(merge_map.items(), key=lambda item: item[1]):
        vocab[idx] = vocab[p0] + vocab[p1]

    return vocab


def validate_specials(specials: Iterable[str]) -> list[str]:
    """Loudly, not helpfully: both of these change the ids silently otherwise.

    Deduping would shift every later special down one; dropping an empty string
    would too, and an empty special matches everywhere in any case. Same reason
    `strict=True` is on in the config loader — a mistake in the recipe should be
    an error, not a vocabulary that isn't the one you asked for.
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
    """Cut `text` into alternating ordinary and special pieces, specials kept.

    This runs *before* the regex split, not after, and that ordering is the whole
    guarantee: a special never reaches `apply_merges`, so no sequence of ordinary
    text can produce its id and no learned merge can be built out of one. Doing
    it the other way round would leave `<|eot|>` as seven characters that a
    sufficiently unlucky corpus could reassemble.

    Longest first, because regex alternation is first-match rather than
    longest-match: with `<|end|>` listed before `<|end|>of<|end|>`, the short one
    would eat the long one's prefix. That ordering is independent of the
    declaration order the ids come from.
    """
    specials = list(specials)
    if not specials:
        return [text]

    pattern = regex.compile(
        "(" + "|".join(regex.escape(s) for s in sorted(specials, key=len, reverse=True)) + ")"
    )
    # `split` on a capturing pattern interleaves the separators into the result,
    # which is why the group is there. The empty strings it puts either side of a
    # match are dropped: they aren't chunks, they're punctuation of the split.
    return [part for part in pattern.split(text) if part]


def apply_merges(tokens: list[int], merge_map: dict[tuple[int, int], int]) -> list[int]:
    """Encode one chunk: repeatedly merge the lowest-ranked pair present.

    Lowest rank, not leftmost or most frequent — rank *is* the order the merges
    were learned in, so replaying them in that order is what reproduces the
    training-time vocabulary. Getting this wrong doesn't crash, it just produces
    plausible-looking ids that disagree with the reference implementation, which
    is why `tests/nanotok` checks against tiktoken rather than eyeballing.
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
