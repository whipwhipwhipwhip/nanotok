"""The byte-level tokenizers.

The load-bearing test in here is `test_matches_tiktoken`. Everything else checks
that our BPE is self-consistent; that one checks it's *right*, against the
reference implementation of the same vocabulary. It's also what makes swapping
`TiktokenBPETokenizer` for a thin `tiktoken` wrapper a config change rather than
a rewrite — the two are interchangeable exactly as long as this passes.
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

# The cases that separate a working BPE from one that looks like it works:
# contractions (GPT-2's pattern is case-sensitive, GPT-4's isn't), digit runs
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
    # Module-scoped, though `_load_encoding` is cached anyway: recovering the
    # merges re-runs BPE over every token in the vocabulary.
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
    # One character, four bytes: the thing that makes byte-level and
    # character-level tokenizers different ID spaces rather than variants.
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
    """`vocab_size` is a ceiling, not a promise — a corpus with no pairs left to
    merge stops short rather than inventing merges."""
    tok = RegexTokenizer()
    tok.train(SPARSE_TEXT, vocab_size=400)

    assert tok.vocab_size < 400


def test_vocab_size_must_exceed_the_byte_tokens():
    with pytest.raises(ValueError, match="256"):
        RegexTokenizer().train(TRAINING_TEXT, vocab_size=100)


def test_merges_do_not_cross_the_split_boundary():
    """The reason the splitter exists: with words and whitespace split apart, no
    learned token can contain both."""
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
    """Trained, with two specials. Module-scoped, so tests that mutate it (by
    retraining) build their own rather than using this one."""
    tok = RegexTokenizer(special_tokens=SPECIALS, split_pattern=r"\S+|\s+")
    tok.train(TRAINING_TEXT, vocab_size=300)
    return tok


def test_specials_are_numbered_above_the_bytes_in_declaration_order():
    tok = RegexTokenizer(special_tokens=SPECIALS)

    assert tok.special_map == {"<|eot|>": 256, "<|pad|>": 257}
    assert tok.vocab_size == 258


def test_declaration_order_decides_the_ids():
    """Order is part of the recipe, not presentation — which is why `to_dict`
    records a list and must never sort it or make it a set."""
    forward = RegexTokenizer(special_tokens=["<|eot|>", "<|pad|>"])
    reversed_ = RegexTokenizer(special_tokens=["<|pad|>", "<|eot|>"])

    assert forward.special_map["<|eot|>"] != reversed_.special_map["<|eot|>"]


def test_an_untrained_tokenizer_already_has_its_specials():
    """Specials are declared, not learned. Registering them inside `train` made
    their ids depend on the corpus and left an untrained instance without them."""
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
    """The split happens *before* the regex, so the text either side of a special
    is chunked and merged exactly as it would be on its own."""
    tok = special_tokenizer
    left, right = "the theory of", " the thing itself"

    assert tok.encode(left + "<|eot|>" + right) == (
        tok.encode(left) + [tok.special_map["<|eot|>"]] + tok.encode(right)
    )


def test_ordinary_text_can_never_forge_a_special_id(special_tokenizer):
    """The reason for splitting before the regex rather than after: a special is
    unreachable from the merge machinery, so no corpus text produces its id."""
    special_ids = set(special_tokenizer.special_map.values())

    assert special_ids
    assert not special_ids & set(special_tokenizer.encode(TRAINING_TEXT))


def test_no_merge_is_built_out_of_a_special(special_tokenizer):
    """Specials never enter `apply_merges`, so no merge pair names one — which is
    also why a missing special leaves a silent hole in a rebuilt vocabulary rather
    than a KeyError."""
    special_ids = set(special_tokenizer.special_map.values())

    assert not any(
        part in special_ids for pair in special_tokenizer.merge_map for part in pair
    )


def test_the_longest_matching_special_wins():
    """Regex alternation is first-match, not longest-match, so the split pattern
    orders by length — independently of the declaration order that fixes the ids.
    Declared shortest-first, the short one would swallow the long one's prefix."""
    tok = RegexTokenizer(special_tokens=["<|end|>", "<|end|>of<|end|>"])

    assert tok.encode("<|end|>of<|end|>") == [257]


def test_without_specials_the_marker_is_just_text():
    """The split step is a no-op with nothing to split on: a tokenizer that
    declares no specials behaves exactly as it did before they existed."""
    tok = RegexTokenizer(split_pattern=r"\S+|\s+")
    tok.train(TRAINING_TEXT, vocab_size=300)

    assert tok.special_map == {}
    assert len(tok.encode("<|eot|>")) > 1
    assert tok.decode(tok.encode("<|eot|>")) == "<|eot|>"


def test_retraining_keeps_the_special_ids_stable():
    """`train` resets the vocabulary before learning, and the reset has to restore
    the specials, not just the 256 bytes — otherwise a second `train` renumbers
    every special and invalidates the checkpoint trained against the first."""
    tok = RegexTokenizer(special_tokens=SPECIALS, split_pattern=r"\S+|\s+")
    tok.train(TRAINING_TEXT, vocab_size=300)
    before = dict(tok.special_map)

    tok.train(TRAINING_TEXT, vocab_size=280)

    assert tok.special_map == before
    assert tok.decode([before["<|eot|>"]]) == "<|eot|>"


def test_vocab_size_must_exceed_the_bytes_plus_the_specials():
    """The ceiling counts specials: two of them leave 258 ids already spoken for,
    so 258 buys no merges at all."""
    tok = RegexTokenizer(special_tokens=SPECIALS)

    with pytest.raises(ValueError, match="258"):
        tok.train(TRAINING_TEXT, vocab_size=258)


def test_a_duplicate_special_is_rejected():
    """Deduping silently would shift every later special down an id, so the
    tokenizer you get is not the one the config describes."""
    with pytest.raises(ValueError, match="distinct"):
        RegexTokenizer(special_tokens=["<|eot|>", "<|pad|>", "<|eot|>"])


def test_an_empty_special_is_rejected():
    """An empty string matches everywhere, so the split would never terminate on
    anything useful — and dropping it quietly renumbers the rest."""
    with pytest.raises(ValueError, match="non-empty"):
        RegexTokenizer(special_tokens=["<|eot|>", ""])


def test_specials_change_the_recipe():
    """`to_dict`'s sha256 is the tokenizer's fingerprint in provenance and part of
    the token cache key. Two tokenizers that encode differently must not serialise
    identically, or the cache serves one's ids for the other."""
    assert RegexTokenizer().to_dict() != RegexTokenizer(
        special_tokens=SPECIALS
    ).to_dict()


def test_specials_survive_a_round_trip_through_to_dict(special_tokenizer):
    """The recipe has to reproduce the whole *id space*, not just the merges. The
    merge ids in `to_dict` are absolute and were numbered above the specials, so
    dropping the specials leaves 256..256+n-1 as holes: `vocab_size` comes back
    short and the checkpoint's embedding table no longer lines up."""
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
    """Our merge loop over their ranks produces their ids. The whole point."""
    assert tiktoken_tokenizer.encode(text) == tiktoken.get_encoding(
        encoding_name
    ).encode(text)


@pytest.mark.parametrize("text", SAMPLES)
def test_tiktoken_round_trip(tiktoken_tokenizer, text):
    assert tiktoken_tokenizer.decode(tiktoken_tokenizer.encode(text)) == text


def test_decodes_merged_ids(tiktoken_tokenizer):
    """Regression: `decode` used to look ids up in a table holding only the 256
    raw bytes, so every merged token — which is to say, almost every token a real
    corpus produces — raised KeyError."""
    ids = tiktoken_tokenizer.encode("hello world")

    assert all(i > 255 for i in ids), "expected merged ids, got raw bytes"
    assert tiktoken_tokenizer.decode(ids) == "hello world"


def test_vocab_size_excludes_special_tokens(tiktoken_tokenizer, encoding_name):
    """Declaring none, we carry the mergeable ranks and nothing else, so we're
    short of tiktoken's `n_vocab` by every id it reserves above them — one for
    GPT-2, twenty-one for cl100k, of which only five are named specials.
    Asserted rather than left implicit because this number sizes the embedding
    table."""
    enc = tiktoken.get_encoding(encoding_name)

    assert tiktoken_tokenizer.vocab_size == len(enc._mergeable_ranks)
    assert tiktoken_tokenizer.vocab_size < enc.n_vocab


def test_gpt2_vocab_size():
    """Pinned separately from the general rule above: 50256-vs-50257 is the
    number that decides whether a GPT-2 checkpoint's embedding table lines up
    with the reference implementation's."""
    assert TiktokenBPETokenizer("gpt2").vocab_size == 50256


def test_training_is_refused(tiktoken_tokenizer):
    with pytest.raises(NotImplementedError, match="RegexTokenizer"):
        tiktoken_tokenizer.train("some text", vocab_size=50300)


def test_tiktoken_survives_a_round_trip_through_to_dict(tiktoken_tokenizer):
    rebuilt = from_dict(tiktoken_tokenizer.to_dict())

    assert rebuilt == tiktoken_tokenizer
    assert rebuilt.encode("hello world") == tiktoken_tokenizer.encode("hello world")


def test_to_dict_records_the_encoding_not_the_merges(tiktoken_tokenizer):
    """A 2MB tokenizer.json per run would be a copy of something that can't
    drift; the name is enough to rebuild."""
    assert "merges" not in tiktoken_tokenizer.to_dict()


###-----------------------------------------------------
# TiktokenBPETokenizer — special tokens


# The one special both encodings define, so the shared tests can run against
# either without knowing which. Its id differs (50256 vs 100257), which is the
# point: the ids come from the encoding, not from a rule of ours.
EOT = "<|endoftext|>"


@pytest.fixture(scope="module")
def special_tiktoken_tokenizer(encoding_name):
    return TiktokenBPETokenizer(encoding_name, special_tokens=[EOT])


def test_a_tiktoken_special_keeps_the_encodings_own_id(
    special_tiktoken_tokenizer, encoding_name
):
    """Not numbered above the ranks the way RegexTokenizer numbers its own: these
    ids already exist, and using different ones would break equivalence."""
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
    """The load-bearing test, extended to specials. `allowed_special` is needed
    on their side because tiktoken refuses to encode a special it wasn't told to
    expect; ours always specialises a declared token, which is the one behavioural
    difference between the two and the reason this test states the set."""
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
    """`decode` un-permutes the whole concatenated byte string at the end, so a
    special stored as its raw utf-8 bytes comes back out scrambled. It has to go
    into `vocab` already in rank space, like everything else in that table."""
    eot = special_tiktoken_tokenizer.special_map[EOT]

    assert special_tiktoken_tokenizer.decode([eot]) == EOT


def test_declaring_endoftext_is_what_makes_gpt2_50257():
    """The open question this answers: a GPT-2 run is 50256 or 50257 depending on
    whether it declared the special, not on a constant chosen once. An id the
    model can never emit is an embedding row trained on nothing."""
    assert TiktokenBPETokenizer("gpt2").vocab_size == 50256
    assert TiktokenBPETokenizer("gpt2", special_tokens=[EOT]).vocab_size == 50257


def test_vocab_size_counts_the_gap_cl100k_leaves():
    """cl100k's specials start at 100257, so 100256 is an id nothing occupies.
    `vocab_size` is highest-id-plus-one rather than a count of entries, because
    an embedding table has to have a row for the gap too."""
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
    """Opting in is what reserves the id. Undeclared, `<|endoftext|>` is thirteen
    characters that merge like any others — where tiktoken would rather raise."""
    ids = tiktoken_tokenizer.encode(EOT)

    assert len(ids) > 1
    assert tiktoken_tokenizer.decode(ids) == EOT


def test_an_unknown_special_is_rejected(encoding_name):
    """We can't mint an id in a vocabulary we didn't train — a name the encoding
    doesn't define has no id to take. RegexTokenizer is where you declare your own."""
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
    """Same reason as for RegexTokenizer: the recipe's sha256 keys the token
    cache, and these two encode the same text differently."""
    assert tiktoken_tokenizer.to_dict() != special_tiktoken_tokenizer.to_dict()


###-----------------------------------------------------
# Vocabulary padding


def round_up(n: int, multiple: int) -> int:
    """What a caller does to arrive at a `pad_vocab_to`. Deliberately here rather
    than in the library: the tokenizer takes the number you want the embedding
    table to be, and how you chose it is your business."""
    return -(-n // multiple) * multiple


@pytest.fixture(scope="module")
def padded_tiktoken_tokenizer(encoding_name):
    """The same encoding, rounded up to a multiple of 64 — 50304 for gpt2,
    100288 for cl100k."""
    unpadded = TiktokenBPETokenizer(encoding_name).vocab_size
    return TiktokenBPETokenizer(
        encoding_name, pad_vocab_to=round_up(unpadded, 64)
    )


@pytest.mark.parametrize("text", SAMPLES)
def test_padding_does_not_touch_the_ids(padded_tiktoken_tokenizer, encoding_name, text):
    """The load-bearing test again, with padding on. Padding adds rows to an
    embedding table; if it can also move an id, it isn't padding, it's a
    different tokenizer wearing the same name."""
    assert padded_tiktoken_tokenizer.encode(text) == tiktoken.get_encoding(
        encoding_name
    ).encode(text)


@pytest.mark.parametrize("text", SAMPLES)
def test_padding_does_not_touch_the_round_trip(padded_tiktoken_tokenizer, text):
    assert (
        padded_tiktoken_tokenizer.decode(padded_tiktoken_tokenizer.encode(text)) == text
    )


def test_gpt2_pads_to_50304():
    """The number this feature exists for. 50257 is 50257; 50304 is 64 x 786, and
    the difference is measurable on the matmuls either side of the model. Pinned
    both with and without the special, because the unpadded size differs and the
    padded one deliberately doesn't."""
    assert TiktokenBPETokenizer("gpt2", pad_vocab_to=50304).vocab_size == 50304
    assert (
        TiktokenBPETokenizer(
            "gpt2", special_tokens=[EOT], pad_vocab_to=50304
        ).vocab_size
        == 50304
    )


def test_padding_leaves_the_unpadded_size_readable(padded_tiktoken_tokenizer):
    """`train` counts with one and the embedding table is sized by the other, so
    a padded tokenizer has to be able to say both."""
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
    """Not `max(unpadded, target)`: the caller stated the size of an embedding
    table, and quietly returning a different number is how a checkpoint ends up
    the wrong shape with nothing to point at."""
    with pytest.raises(ValueError, match="below the 50256 ids"):
        TiktokenBPETokenizer("gpt2", pad_vocab_to=50000).vocab_size


@pytest.mark.parametrize("bad", [0, -64, 1.5, "50304"])
def test_padding_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError, match="positive integer"):
        ByteTokenizer(pad_vocab_to=bad).vocab_size


def test_a_padding_id_decodes_to_nothing(padded_tiktoken_tokenizer):
    """Inside `vocab_size` and outside `vocab` — the same shape of object as the
    hole cl100k leaves at 100256. The message says "padding" rather than
    "outside" because the id *is* inside the vocabulary, and the actual bug is a
    sampler drawing from logits nobody masked."""
    pad_id = padded_tiktoken_tokenizer.unpadded_vocab_size

    assert pad_id < padded_tiktoken_tokenizer.vocab_size

    with pytest.raises(ValueError, match="padding"):
        padded_tiktoken_tokenizer.decode([pad_id])


def test_an_id_above_the_padding_is_still_out_of_range(padded_tiktoken_tokenizer):
    with pytest.raises(ValueError, match="outside"):
        padded_tiktoken_tokenizer.decode([padded_tiktoken_tokenizer.vocab_size])


def test_padding_changes_the_recipe(tiktoken_tokenizer, padded_tiktoken_tokenizer):
    """Same reason specials do: the fingerprint keys the token cache and goes
    into a run's provenance, and these two size different embedding tables."""
    assert tiktoken_tokenizer.to_dict() != padded_tiktoken_tokenizer.to_dict()
    assert fingerprint(tiktoken_tokenizer) != fingerprint(padded_tiktoken_tokenizer)


def test_an_unpadded_recipe_records_no_padding(tiktoken_tokenizer):
    """The key is omitted rather than written as null, so every fingerprint taken
    before padding existed still identifies the same tokenizer."""
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
    """`train`'s ceiling asks for merges to be learned. Padding adds ids no merge
    can reach, so counting it would silently buy fewer merges than were asked
    for — 44 instead of 300 here."""
    tok = RegexTokenizer(split_pattern=r"\S+|\s+", pad_vocab_to=512)
    tok.train(TRAINING_TEXT, vocab_size=300)

    assert len(tok.vocab) == 300
    assert tok.unpadded_vocab_size == 300
    assert tok.vocab_size == 512


def test_padding_survives_training():
    """Padding is config, not learned state, so the reset at the top of `train`
    must not take it with the merges."""
    tok = RegexTokenizer(pad_vocab_to=512)
    tok.train(TRAINING_TEXT, vocab_size=300)
    tok.train(TRAINING_TEXT, vocab_size=280)

    assert tok.vocab_size == 512


def test_training_past_the_padding_is_rejected():
    """Caught when the size is asked for rather than silently ignored: a padded
    tokenizer trained past its target has an embedding table sized by neither
    number, and the checkpoint is the thing that finds out."""
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
