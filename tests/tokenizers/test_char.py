"""CharTokenizer: the id assignment, and the two ways of choosing an alphabet.

Nothing here touches the filesystem or torch. The interesting property is not
"encode then decode gives the text back" — it's that the *same* alphabet always
produces the *same* ids, because a checkpoint is only readable if that holds.
"""

from __future__ import annotations

import pytest

from tokenizers import CharTokenizer, fingerprint, from_dict


def test_round_trip():
    tok = CharTokenizer.from_text("hello world")
    assert tok.decode(tok.encode("hello world")) == "hello world"


def test_vocab_is_the_distinct_characters():
    tok = CharTokenizer.from_text("hello world")
    # h e l o w r d and a space
    assert tok.vocab_size == 8
    assert tok.alphabet == " dehlorw"


def test_ids_do_not_depend_on_the_order_characters_arrive_in():
    """The property a fixed alphabet exists to give: `from_alphabet("+0123")` and
    `from_alphabet("3210+")` are the same tokenizer, so a config that lists the
    adder's characters in a different order doesn't silently renumber them."""
    assert CharTokenizer.from_alphabet("0123456789+=\n") == CharTokenizer.from_alphabet(
        "\n=+9876543210"
    )


def test_duplicate_characters_are_ignored():
    assert CharTokenizer.from_alphabet("aabbc").alphabet == "abc"


def test_fixed_alphabet_does_not_depend_on_the_corpus():
    """The reason `from_alphabet` exists: a corpus that happens not to contain a
    '9' still gets a tokenizer that can represent one."""
    tok = CharTokenizer.from_alphabet("0123456789+=\n")
    assert tok.vocab_size == 13
    assert tok.decode(tok.encode("12+9=21\n")) == "12+9=21\n"


def test_unknown_character_names_itself():
    tok = CharTokenizer.from_alphabet("0123456789+=\n")

    with pytest.raises(ValueError, match="'x'"):
        tok.encode("12x9")


def test_unknown_id_is_rejected():
    tok = CharTokenizer.from_alphabet("abc")

    with pytest.raises(ValueError, match="outside"):
        tok.decode([0, 99])


def test_empty_alphabet_is_rejected():
    with pytest.raises(ValueError, match="empty alphabet"):
        CharTokenizer.from_text("")


def test_survives_a_round_trip_through_to_dict():
    tok = CharTokenizer.from_text("hello world")
    assert from_dict(tok.to_dict()) == tok


def test_to_dict_is_json_shaped():
    tok = CharTokenizer.from_alphabet("ab")
    assert tok.to_dict() == {"kind": "char", "alphabet": "ab"}


def test_padding_rounds_the_vocabulary_up_without_adding_characters():
    """The adder's thirteen characters in an embedding table of 16 rows. The
    three extra ids stand for nothing and are never emitted — see
    `tokenizers.base.pad_vocab_size`."""
    tok = CharTokenizer.from_alphabet("0123456789+=\n", pad_vocab_to=16)

    assert tok.vocab_size == 16
    assert tok.unpadded_vocab_size == 13
    assert tok.alphabet == "\n+0123456789="
    assert tok.encode("12+9=21\n") == CharTokenizer.from_alphabet(
        "0123456789+=\n"
    ).encode("12+9=21\n")


def test_a_padding_id_decodes_to_nothing():
    tok = CharTokenizer.from_alphabet("abc", pad_vocab_to=8)

    with pytest.raises(ValueError, match="padding"):
        tok.decode([3])


def test_padding_below_the_alphabet_is_rejected():
    with pytest.raises(ValueError, match="below the 3 ids"):
        CharTokenizer.from_alphabet("abc", pad_vocab_to=2).vocab_size


def test_padding_makes_a_different_tokenizer():
    """Frozen dataclass equality has to see it: same alphabet, different
    embedding table, so a checkpoint from one won't load into the other."""
    assert CharTokenizer.from_alphabet("abc") != CharTokenizer.from_alphabet(
        "abc", pad_vocab_to=8
    )
    assert fingerprint(CharTokenizer.from_alphabet("abc")) != fingerprint(
        CharTokenizer.from_alphabet("abc", pad_vocab_to=8)
    )


def test_padding_survives_a_round_trip_through_to_dict():
    tok = CharTokenizer.from_text("hello world", pad_vocab_to=16)
    rebuilt = from_dict(tok.to_dict())

    assert rebuilt == tok
    assert rebuilt.vocab_size == 16


def test_padded_to_dict_is_json_shaped():
    tok = CharTokenizer.from_alphabet("ab", pad_vocab_to=8)
    assert tok.to_dict() == {"kind": "char", "alphabet": "ab", "pad_vocab_to": 8}
