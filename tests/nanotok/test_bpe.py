"""The byte-level tokenizers.

The most important test here is `test_matches_tiktoken`. Most of the others
check that our BPE is self-consistent; that one checks it gives the same ids as
tiktoken for the same vocabulary. As long as it passes, `TiktokenBPETokenizer`
could be swapped for a plain `tiktoken` wrapper.
"""

from __future__ import annotations

import pytest
import tiktoken

from nanotok import (
    ByteTokenizer,
    RegexTokenizer,
    TiktokenBPETokenizer,
    fingerprint,
    from_dict,
)

# Cases where it's easy to get BPE subtly wrong: contractions (GPT-2's pattern is case-sensitive, GPT-4's isn't), digit runs
# (GPT-4 caps them at three), leading and trailing whitespace, repeated
# whitespace, and multi-byte characters that a byte-level merge can split.
SAMPLES = [
    "",
    "hello world",
    "don't I'LL we've they're",
    "Hello   World",
    "1234567890",
    "héllo wörld 🙂",
    "a\n",
    "hello\n\n",
    "  leading",
    "trailing  ",
    "tab\there",
    "The quick brown fox jumps over the lazy dog. 42 times!",
]

# Repetitive enough that BPE has plenty to find, and long enough to reach a
# 300-token vocabulary before it runs out of repeated pairs.
TRAINING_TEXT = (
    "The theory of the theatre, then, is the theory of the thing itself. "
    "Whatever the players think they are doing, the audience thinks otherwise, "
    "and the difference between those two thoughts is where the evening lives. "
    "A theatre without that difference is a lecture with better lighting. "
) * 20

# Short and unrepetitive: nothing left to merge long before the ceiling.
SPARSE_TEXT = "abcdef"


@pytest.fixture(scope="module", params=["gpt2", "cl100k_base"])
def encoding_name(request):
    return request.param


@pytest.fixture(scope="module")
def tiktoken_tokenizer(encoding_name):
    # Module-scoped (though `_load_encoding` is cached anyway), since
    # recovering the merges re-runs BPE over every token in the vocabulary.
    return TiktokenBPETokenizer(encoding_name)


###-----------------------------------------------------
# ByteTokenizer


def test_byte_vocab_is_the_256_bytes():
    assert ByteTokenizer().vocab_size == 256


@pytest.mark.parametrize("text", SAMPLES)
def test_byte_round_trip(text):
    tok = ByteTokenizer()
    assert tok.decode(tok.encode(text)) == text


def test_byte_encodes_multibyte_characters_as_several_ids():
    # One character, four bytes.
    assert len(ByteTokenizer().encode("🙂")) == 4


def test_byte_rejects_an_id_it_never_emits():
    with pytest.raises(ValueError, match="outside"):
        ByteTokenizer().decode([256])


###-----------------------------------------------------
# RegexTokenizer


def test_untrained_regex_tokenizer_is_a_byte_tokenizer():
    """Regression: `vocab_size` used to be set inside `train`, so an untrained
    instance raised AttributeError instead of answering 256."""
    tok = RegexTokenizer()
    assert tok.vocab_size == 256
    assert tok.encode("hello") == ByteTokenizer().encode("hello")


def test_training_grows_the_vocabulary_and_shortens_the_encoding():
    tok = RegexTokenizer()
    tok.train(TRAINING_TEXT, vocab_size=300)

    assert tok.vocab_size == 300
    assert len(tok.encode(TRAINING_TEXT)) < len(ByteTokenizer().encode(TRAINING_TEXT))
    assert tok.decode(tok.encode(TRAINING_TEXT)) == TRAINING_TEXT


def test_training_stops_early_when_nothing_repeats():
    """`vocab_size` is an upper limit. If no pair repeats, training stops
    early."""
    tok = RegexTokenizer()
    tok.train(SPARSE_TEXT, vocab_size=400)

    assert tok.vocab_size < 400


def test_vocab_size_must_exceed_the_byte_tokens():
    with pytest.raises(ValueError, match="256"):
        RegexTokenizer().train(TRAINING_TEXT, vocab_size=100)


def test_merges_do_not_cross_the_split_boundary():
    """With words and whitespace split apart, no learned token can contain
    both."""
    tok = RegexTokenizer(split_pattern=r"\S+|\s+")
    tok.train("aa bb aa bb aa bb ", vocab_size=270)

    for token_bytes in tok.vocab.values():
        text = token_bytes.decode("utf-8", errors="replace")
        assert not (text.strip() and text != text.strip()), (
            f"token {text!r} spans a whitespace boundary"
        )


@pytest.mark.parametrize("text", SAMPLES)
def test_trained_regex_round_trip(text):
    tok = RegexTokenizer()
    tok.train(TRAINING_TEXT, vocab_size=300)
    assert tok.decode(tok.encode(text)) == text


def test_trained_regex_survives_a_round_trip_through_to_dict():
    tok = RegexTokenizer(split_pattern=r"\S+|\s+")
    tok.train(TRAINING_TEXT, vocab_size=300)

    rebuilt = from_dict(tok.to_dict())

    assert rebuilt == tok
    assert rebuilt.vocab == tok.vocab
    assert rebuilt.encode(TRAINING_TEXT) == tok.encode(TRAINING_TEXT)


###-----------------------------------------------------
# RegexTokenizer — special tokens


SPECIALS = ["<|eot|>", "<|pad|>"]


@pytest.fixture(scope="module")
def special_tokenizer():
    """Trained, with two specials. Module-scoped, so tests that retrain should
    build their own instead."""
    tok = RegexTokenizer(special_tokens=SPECIALS, split_pattern=r"\S+|\s+")
    tok.train(TRAINING_TEXT, vocab_size=300)
    return tok


def test_specials_are_numbered_above_the_bytes_in_declaration_order():
    tok = RegexTokenizer(special_tokens=SPECIALS)

    assert tok.special_map == {"<|eot|>": 256, "<|pad|>": 257}
    assert tok.vocab_size == 258


def test_declaration_order_decides_the_ids():
    """Order matters, which is why `to_dict` saves a list and must never sort
    it or turn it into a set."""
    forward = RegexTokenizer(special_tokens=["<|eot|>", "<|pad|>"])
    reversed_ = RegexTokenizer(special_tokens=["<|pad|>", "<|eot|>"])

    assert forward.special_map["<|eot|>"] != reversed_.special_map["<|eot|>"]


def test_an_untrained_tokenizer_already_has_its_specials():
    """Specials are declared, not learned, so they exist before `train` is
    called."""
    tok = RegexTokenizer(special_tokens=["<|eot|>"])

    assert tok.encode("<|eot|>") == [256]
    assert tok.decode([256]) == "<|eot|>"


def test_a_special_encodes_to_exactly_one_id(special_tokenizer):
    eot = special_tokenizer.special_map["<|eot|>"]

    assert special_tokenizer.encode("<|eot|>") == [eot]


@pytest.mark.parametrize(
    "text",
    [
        "<|eot|>",
        "<|eot|>the theory",
        "the theory<|eot|>",
        "the<|eot|>theory",
        "<|eot|><|pad|>",
        "<|eot|> <|pad|>",
        "a<|eot|>b<|pad|>c",
        "<|eot|>" * 3,
    ],
)
def test_special_round_trip(special_tokenizer, text):
    assert special_tokenizer.decode(special_tokenizer.encode(text)) == text


@pytest.mark.parametrize("text", SAMPLES)
def test_declaring_specials_leaves_ordinary_text_round_tripping(
    special_tokenizer, text
):
    assert special_tokenizer.decode(special_tokenizer.encode(text)) == text


def test_a_special_does_not_disturb_its_neighbours(special_tokenizer):
    """The split on specials happens before the regex, so the text either side
    of a special encodes the same as it would on its own."""
    tok = special_tokenizer
    left, right = "the theory of", " the thing itself"

    assert tok.encode(left + "<|eot|>" + right) == (
        tok.encode(left) + [tok.special_map["<|eot|>"]] + tok.encode(right)
    )


def test_ordinary_text_can_never_forge_a_special_id(special_tokenizer):
    """Specials never go through the merges, so no ordinary text can produce
    a special's id."""
    special_ids = set(special_tokenizer.special_map.values())

    assert special_ids
    assert not special_ids & set(special_tokenizer.encode(TRAINING_TEXT))


def test_no_merge_is_built_out_of_a_special(special_tokenizer):
    """Specials never go through `apply_merges`, so no merge pair contains
    one."""
    special_ids = set(special_tokenizer.special_map.values())

    assert not any(
        part in special_ids for pair in special_tokenizer.merge_map for part in pair
    )


def test_the_longest_matching_special_wins():
    """Regex alternation takes the first match, not the longest, so the split
    pattern is ordered by length (the ids still follow declaration order).
    Otherwise the short one would match the start of the long one."""
    tok = RegexTokenizer(special_tokens=["<|end|>", "<|end|>of<|end|>"])

    assert tok.encode("<|end|>of<|end|>") == [257]


def test_without_specials_the_marker_is_just_text():
    """With no specials declared, `<|eot|>` is just ordinary text."""
    tok = RegexTokenizer(split_pattern=r"\S+|\s+")
    tok.train(TRAINING_TEXT, vocab_size=300)

    assert tok.special_map == {}
    assert len(tok.encode("<|eot|>")) > 1
    assert tok.decode(tok.encode("<|eot|>")) == "<|eot|>"


def test_retraining_keeps_the_special_ids_stable():
    """`train` resets the vocabulary first, and the reset has to put the
    specials back with the same ids, or a checkpoint trained with the old ids
    would break."""
    tok = RegexTokenizer(special_tokens=SPECIALS, split_pattern=r"\S+|\s+")
    tok.train(TRAINING_TEXT, vocab_size=300)
    before = dict(tok.special_map)

    tok.train(TRAINING_TEXT, vocab_size=280)

    assert tok.special_map == before
    assert tok.decode([before["<|eot|>"]]) == "<|eot|>"


def test_vocab_size_must_exceed_the_bytes_plus_the_specials():
    """`vocab_size` counts the specials: with two of them, 258 ids are already
    taken, so 258 would mean no merges."""
    tok = RegexTokenizer(special_tokens=SPECIALS)

    with pytest.raises(ValueError, match="258"):
        tok.train(TRAINING_TEXT, vocab_size=258)


def test_a_duplicate_special_is_rejected():
    """Silently removing the duplicate would shift every later special down an
    id."""
    with pytest.raises(ValueError, match="distinct"):
        RegexTokenizer(special_tokens=["<|eot|>", "<|pad|>", "<|eot|>"])


def test_an_empty_special_is_rejected():
    """An empty string matches everywhere, and silently dropping it would
    renumber the rest."""
    with pytest.raises(ValueError, match="non-empty"):
        RegexTokenizer(special_tokens=["<|eot|>", ""])


def test_specials_change_the_recipe():
    """`to_dict` is hashed for the tokenizer's fingerprint, so two tokenizers
    that encode differently must not serialise the same."""
    assert RegexTokenizer().to_dict() != RegexTokenizer(
        special_tokens=SPECIALS
    ).to_dict()


def test_specials_survive_a_round_trip_through_to_dict(special_tokenizer):
    """The merge ids in `to_dict` are numbered after the specials, so if the
    specials weren't saved, the rebuilt vocabulary would have gaps at 256.."""
    rebuilt = from_dict(special_tokenizer.to_dict())

    assert rebuilt == special_tokenizer
    assert rebuilt.special_map == special_tokenizer.special_map
    assert rebuilt.vocab == special_tokenizer.vocab
    assert rebuilt.vocab_size == special_tokenizer.vocab_size
    assert rebuilt.decode([256]) == "<|eot|>"
    assert rebuilt.encode("a<|eot|>b") == special_tokenizer.encode("a<|eot|>b")


###-----------------------------------------------------
# TiktokenBPETokenizer


@pytest.mark.parametrize("text", SAMPLES)
def test_matches_tiktoken(tiktoken_tokenizer, encoding_name, text):
    """Our merge loop over tiktoken's ranks gives tiktoken's ids."""
    assert tiktoken_tokenizer.encode(text) == tiktoken.get_encoding(
        encoding_name
    ).encode(text)


@pytest.mark.parametrize("text", SAMPLES)
def test_tiktoken_round_trip(tiktoken_tokenizer, text):
    assert tiktoken_tokenizer.decode(tiktoken_tokenizer.encode(text)) == text


def test_decodes_merged_ids(tiktoken_tokenizer):
    """Regression: `decode` used to look ids up in a table with only the 256
    raw bytes, so any merged token raised KeyError."""
    ids = tiktoken_tokenizer.encode("hello world")

    assert all(i > 255 for i in ids), "expected merged ids, got raw bytes"
    assert tiktoken_tokenizer.decode(ids) == "hello world"


def test_vocab_size_excludes_special_tokens(tiktoken_tokenizer, encoding_name):
    """With no specials declared, we only have the mergeable ranks, so we're
    smaller than tiktoken's `n_vocab` by every id it reserves above them: one
    for GPT-2, twenty-one for cl100k (of which only five are named specials)."""
    enc = tiktoken.get_encoding(encoding_name)

    assert tiktoken_tokenizer.vocab_size == len(enc._mergeable_ranks)
    assert tiktoken_tokenizer.vocab_size < enc.n_vocab


def test_gpt2_vocab_size():
    """Checked explicitly because 50256 vs 50257 decides whether the embedding
    table matches GPT-2's."""
    assert TiktokenBPETokenizer("gpt2").vocab_size == 50256


def test_training_is_refused(tiktoken_tokenizer):
    with pytest.raises(NotImplementedError, match="RegexTokenizer"):
        tiktoken_tokenizer.train("some text", vocab_size=50300)


def test_tiktoken_survives_a_round_trip_through_to_dict(tiktoken_tokenizer):
    rebuilt = from_dict(tiktoken_tokenizer.to_dict())

    assert rebuilt == tiktoken_tokenizer
    assert rebuilt.encode("hello world") == tiktoken_tokenizer.encode("hello world")


def test_to_dict_records_the_encoding_not_the_merges(tiktoken_tokenizer):
    """The encoding name is enough to rebuild it."""
    assert "merges" not in tiktoken_tokenizer.to_dict()


###-----------------------------------------------------
# TiktokenBPETokenizer — special tokens


# The one special both encodings define, so the shared tests can run against
# either. Its id differs (50256 vs 100257) because the ids come from the
# encoding.
EOT = "<|endoftext|>"


@pytest.fixture(scope="module")
def special_tiktoken_tokenizer(encoding_name):
    return TiktokenBPETokenizer(encoding_name, special_tokens=[EOT])


def test_a_tiktoken_special_keeps_the_encodings_own_id(
    special_tiktoken_tokenizer, encoding_name
):
    """Unlike RegexTokenizer, the ids come from the encoding, so the output
    still matches tiktoken."""
    enc = tiktoken.get_encoding(encoding_name)

    assert special_tiktoken_tokenizer.special_map == {EOT: enc._special_tokens[EOT]}


@pytest.mark.parametrize(
    "text",
    [
        EOT,
        f"hello{EOT}world",
        f"{EOT}hello",
        f"hello{EOT}",
        f"{EOT}{EOT}",
        f"a {EOT} b",
    ],
)
def test_matches_tiktoken_with_specials(
    special_tiktoken_tokenizer, encoding_name, text
):
    """`test_matches_tiktoken` with specials. tiktoken needs `allowed_special`
    or it refuses to encode a special, whereas ours always treats a declared
    special as special. That's the one difference in behaviour between them."""
    assert special_tiktoken_tokenizer.encode(text) == tiktoken.get_encoding(
        encoding_name
    ).encode(text, allowed_special={EOT})


@pytest.mark.parametrize("text", [*SAMPLES, f"hello{EOT}world"])
def test_tiktoken_special_round_trip(special_tiktoken_tokenizer, text):
    assert special_tiktoken_tokenizer.decode(
        special_tiktoken_tokenizer.encode(text)
    ) == text


def test_a_tiktoken_special_decodes_back_through_the_byte_shuffle(
    special_tiktoken_tokenizer,
):
    """`decode` un-shuffles all the bytes at the end, so a special has to be
    stored in `vocab` in rank space, or it would come out scrambled."""
    eot = special_tiktoken_tokenizer.special_map[EOT]

    assert special_tiktoken_tokenizer.decode([eot]) == EOT


def test_declaring_endoftext_is_what_makes_gpt2_50257():
    """GPT-2's vocab size is 50256 or 50257 depending on whether the special
    is declared."""
    assert TiktokenBPETokenizer("gpt2").vocab_size == 50256
    assert TiktokenBPETokenizer("gpt2", special_tokens=[EOT]).vocab_size == 50257


def test_vocab_size_counts_the_gap_cl100k_leaves():
    """cl100k's specials start at 100257, so 100256 is unused. `vocab_size` is
    the highest id plus one rather than a count of entries, because the
    embedding table still needs a row for the gap."""
    tok = TiktokenBPETokenizer("cl100k_base", special_tokens=[EOT])

    assert tok.special_map[EOT] == 100257
    assert tok.vocab_size == 100258
    assert len(tok.vocab) == 100257  # one short of vocab_size: 100256 is unused

    with pytest.raises(ValueError, match="outside"):
        tok.decode([100256])


def test_declaring_every_cl100k_special_reaches_n_vocab():
    enc = tiktoken.get_encoding("cl100k_base")
    tok = TiktokenBPETokenizer(
        "cl100k_base", special_tokens=sorted(enc._special_tokens)
    )

    assert tok.vocab_size == enc.n_vocab


def test_an_undeclared_special_is_ordinary_text(tiktoken_tokenizer):
    """Undeclared, `<|endoftext|>` is just thirteen characters that merge like
    any others (where tiktoken would raise)."""
    ids = tiktoken_tokenizer.encode(EOT)

    assert len(ids) > 1
    assert tiktoken_tokenizer.decode(ids) == EOT


def test_an_unknown_special_is_rejected(encoding_name):
    """A name the encoding doesn't define has no id to use. Use RegexTokenizer
    to define your own."""
    with pytest.raises(ValueError, match="no special token"):
        TiktokenBPETokenizer(encoding_name, special_tokens=["<|pad|>"])


def test_tiktoken_specials_survive_a_round_trip_through_to_dict(
    special_tiktoken_tokenizer,
):
    rebuilt = from_dict(special_tiktoken_tokenizer.to_dict())

    assert rebuilt == special_tiktoken_tokenizer
    assert rebuilt.special_map == special_tiktoken_tokenizer.special_map
    assert rebuilt.vocab_size == special_tiktoken_tokenizer.vocab_size
    assert rebuilt.encode(f"a{EOT}b") == special_tiktoken_tokenizer.encode(f"a{EOT}b")


def test_specials_change_the_tiktoken_recipe(
    tiktoken_tokenizer, special_tiktoken_tokenizer
):
    """Same as for RegexTokenizer: these two encode differently, so their
    recipes (and fingerprints) must differ."""
    assert tiktoken_tokenizer.to_dict() != special_tiktoken_tokenizer.to_dict()


###-----------------------------------------------------
# Vocabulary padding


def round_up(n: int, multiple: int) -> int:
    """How a caller might pick a `pad_vocab_to`. Not part of the library, which
    just takes the final size."""
    return -(-n // multiple) * multiple


@pytest.fixture(scope="module")
def padded_tiktoken_tokenizer(encoding_name):
    """The same encoding, rounded up to a multiple of 64 (50304 for gpt2,
    100288 for cl100k)."""
    unpadded = TiktokenBPETokenizer(encoding_name).vocab_size
    return TiktokenBPETokenizer(
        encoding_name, pad_vocab_to=round_up(unpadded, 64)
    )


@pytest.mark.parametrize("text", SAMPLES)
def test_padding_does_not_touch_the_ids(padded_tiktoken_tokenizer, encoding_name, text):
    """`test_matches_tiktoken` again, with padding on. Padding should only
    change `vocab_size`, never the ids."""
    assert padded_tiktoken_tokenizer.encode(text) == tiktoken.get_encoding(
        encoding_name
    ).encode(text)


@pytest.mark.parametrize("text", SAMPLES)
def test_padding_does_not_touch_the_round_trip(padded_tiktoken_tokenizer, text):
    assert (
        padded_tiktoken_tokenizer.decode(padded_tiktoken_tokenizer.encode(text)) == text
    )


def test_gpt2_pads_to_50304():
    """The main use case: 50304 = 64 x 786. Checked with and without the
    special, since the unpadded size differs but the padded size shouldn't."""
    assert TiktokenBPETokenizer("gpt2", pad_vocab_to=50304).vocab_size == 50304
    assert (
        TiktokenBPETokenizer(
            "gpt2", special_tokens=[EOT], pad_vocab_to=50304
        ).vocab_size
        == 50304
    )


def test_padding_leaves_the_unpadded_size_readable(padded_tiktoken_tokenizer):
    """`train` uses the unpadded size and the embedding table uses the padded
    one, so both need to be available."""
    assert padded_tiktoken_tokenizer.unpadded_vocab_size < (
        padded_tiktoken_tokenizer.vocab_size
    )
    assert padded_tiktoken_tokenizer.unpadded_vocab_size == len(
        padded_tiktoken_tokenizer.vocab
    )


def test_padding_to_the_size_it_already_is_changes_nothing():
    assert TiktokenBPETokenizer("gpt2", pad_vocab_to=50256).vocab_size == 50256
    assert ByteTokenizer(pad_vocab_to=256).vocab_size == 256


def test_padding_below_the_vocabulary_is_rejected():
    """Raises rather than returning `max(unpadded, target)`, since the caller
    asked for a specific embedding table size."""
    with pytest.raises(ValueError, match="below the 50256 ids"):
        TiktokenBPETokenizer("gpt2", pad_vocab_to=50000).vocab_size


@pytest.mark.parametrize("bad", [0, -64, 1.5, "50304"])
def test_padding_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError, match="positive integer"):
        ByteTokenizer(pad_vocab_to=bad).vocab_size


def test_a_padding_id_decodes_to_nothing(padded_tiktoken_tokenizer):
    """A padding id is inside `vocab_size` but not in `vocab`, like the gap
    cl100k leaves at 100256. The error says "padding" rather than "outside",
    since the usual cause is a sampler that didn't mask the padding logits."""
    pad_id = padded_tiktoken_tokenizer.unpadded_vocab_size

    assert pad_id < padded_tiktoken_tokenizer.vocab_size

    with pytest.raises(ValueError, match="padding"):
        padded_tiktoken_tokenizer.decode([pad_id])


def test_an_id_above_the_padding_is_still_out_of_range(padded_tiktoken_tokenizer):
    with pytest.raises(ValueError, match="outside"):
        padded_tiktoken_tokenizer.decode([padded_tiktoken_tokenizer.vocab_size])


def test_padding_changes_the_recipe(tiktoken_tokenizer, padded_tiktoken_tokenizer):
    """These two need different embedding tables, so their recipes (and
    fingerprints) must differ."""
    assert tiktoken_tokenizer.to_dict() != padded_tiktoken_tokenizer.to_dict()
    assert fingerprint(tiktoken_tokenizer) != fingerprint(padded_tiktoken_tokenizer)


def test_an_unpadded_recipe_records_no_padding(tiktoken_tokenizer):
    """The key is left out rather than written as null, so fingerprints from
    before padding was added still match."""
    assert "pad_vocab_to" not in tiktoken_tokenizer.to_dict()
    assert "pad_vocab_to" not in RegexTokenizer().to_dict()
    assert ByteTokenizer().to_dict() == {"kind": "byte"}


@pytest.mark.parametrize(
    "tokenizer",
    [
        ByteTokenizer(pad_vocab_to=320),
        RegexTokenizer(special_tokens=SPECIALS, pad_vocab_to=512),
        TiktokenBPETokenizer("gpt2", special_tokens=[EOT], pad_vocab_to=50304),
    ],
    ids=["byte", "regex", "tiktoken"],
)
def test_padding_survives_a_round_trip_through_to_dict(tokenizer):
    rebuilt = from_dict(tokenizer.to_dict())

    assert rebuilt == tokenizer
    assert rebuilt.vocab_size == tokenizer.vocab_size
    assert rebuilt.pad_vocab_to == tokenizer.pad_vocab_to


def test_the_training_target_ignores_the_padding():
    """`train`'s `vocab_size` doesn't count padding. If it did, this would only
    learn 44 merges instead of 300."""
    tok = RegexTokenizer(split_pattern=r"\S+|\s+", pad_vocab_to=512)
    tok.train(TRAINING_TEXT, vocab_size=300)

    assert len(tok.vocab) == 300
    assert tok.unpadded_vocab_size == 300
    assert tok.vocab_size == 512


def test_padding_survives_training():
    """Padding is config, not learned state, so the reset at the start of
    `train` shouldn't remove it."""
    tok = RegexTokenizer(pad_vocab_to=512)
    tok.train(TRAINING_TEXT, vocab_size=300)
    tok.train(TRAINING_TEXT, vocab_size=280)

    assert tok.vocab_size == 512


def test_training_past_the_padding_is_rejected():
    """If training goes past the padded size, asking for `vocab_size` raises
    rather than silently returning the wrong number."""
    tok = RegexTokenizer(pad_vocab_to=280)
    tok.train(TRAINING_TEXT, vocab_size=300)

    with pytest.raises(ValueError, match="below the 300 ids"):
        tok.vocab_size


###-----------------------------------------------------
# The dispatcher


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown tokenizer kind"):
        from_dict({"kind": "no_such_tokenizer"})


def test_missing_kind_is_rejected():
    with pytest.raises(ValueError, match="no 'kind' key"):
        from_dict({"alphabet": "abc"})
