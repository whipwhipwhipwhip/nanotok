## Nanotok

This is my collection of tokenizer code I made while following Karpathy's Zero-to-hero lectures and then doing my own GPT experiments.

It started as the code from [Let's build the GPT tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE), and grew into a small package so my GPT experiments could all use the same tokenizers. The follow-along notebooks for the lecture are in my [zero-to-hero](https://github.com/whipwhipwhipwhip/zero-to-hero/tree/main/notebooks/05_tokenizers) repo.

It's not meant to replace `tiktoken` (which is much faster, since it does the merging in Rust). The goal was to write BPE myself and check that it gives the same answer.

### What's here

There are four tokenizers, all in [src/nanotok](src/nanotok), and they all have the same interface (`encode`, `decode`, `vocab_size`, `to_dict` / `from_dict`):

- [char.py](src/nanotok/char.py) - `CharTokenizer`, one id per character, as in the nanoGPT lecture. `from_text` gets the alphabet from a corpus. `from_alphabet` takes it as an argument, which is better for tasks with a fixed set of characters (e.g. `"0123456789+=\n"` for an addition task), because the ids then don't change when the data does.
- [byte.py](src/nanotok/byte.py) - `ByteTokenizer`, one id per UTF-8 byte, so a vocabulary of 256 and nothing to train. It's also the base class for the two BPE tokenizers (BPE with no merges is just this), and it handles special tokens for both of them.
- [regex_bpe.py](src/nanotok/regex_bpe.py) - `RegexTokenizer`, BPE trained from scratch. The text is split with a regex first, so merges can't cross chunk boundaries (e.g. joining a word to the space after it). By default the whole text is one chunk, which is the "basic" tokenizer from the lecture.
- [tiktoken_bpe.py](src/nanotok/tiktoken_bpe.py) - `TiktokenBPETokenizer`, my merge loop running on the vocabulary of a real tiktoken encoding (`gpt2` or `cl100k_base`). This is the lecture's GPT-4 exercise. tiktoken only stores the merged byte strings, not which pair made each one, so `recover_merges` re-runs BPE on every token to get the pairs back.
- [merges.py](src/nanotok/merges.py) - the BPE functions both BPE tokenizers use: counting pairs, merging, encoding a chunk, building the vocabulary, and splitting out special tokens.

Beyond the lecture, I added a few things I needed for training models:

- **Special tokens** like `<|endoftext|>` are declared when you build the tokenizer, and never learned. `RegexTokenizer` gives them the ids right after the 256 bytes. `TiktokenBPETokenizer` uses the ids the encoding already has (50256 for GPT-2), so its output still matches `tiktoken.encode`.
- **Vocabulary padding**. `pad_vocab_to` rounds `vocab_size` up to a number that's faster on a GPU, e.g. GPT-2's 50257 up to 50304 (= 64 x 786). The extra ids are never produced by `encode` and nothing else changes.
- **Saving and loading**. `to_dict()` returns a small JSON-able description of how to rebuild the tokenizer (an encoding name, an alphabet, or a list of merges), and `nanotok.from_dict` rebuilds it. `nanotok.fingerprint` is the sha256 of that description, which I use to record which tokenizer a training run used.

### Usage

```python
from nanotok import RegexTokenizer, TiktokenBPETokenizer, from_dict

tok = TiktokenBPETokenizer("gpt2", special_tokens=["<|endoftext|>"], pad_vocab_to=50304)
tok.encode("hello world<|endoftext|>")  # [31373, 995, 50256], same as tiktoken
tok.vocab_size                          # 50304

bpe = RegexTokenizer(special_tokens=["<|eot|>"])
bpe.train(open("input.txt").read(), vocab_size=512)
bpe = from_dict(bpe.to_dict())          # rebuilt from its saved recipe
```

The first `TiktokenBPETokenizer` for an encoding takes a few seconds to build, since it has to recover all the merges (after that it's cached).

### Tests

```bash
uv run pytest
```

The main test is `test_matches_tiktoken` in [test_bpe.py](tests/nanotok/test_bpe.py), which checks that `TiktokenBPETokenizer` gives exactly the same ids as `tiktoken` for both `gpt2` and `cl100k_base`. The test strings are cases where it's easy to get BPE subtly wrong: contractions, long runs of digits, repeated whitespace, and multi-byte characters. The rest of the tests check round trips, training, special tokens, padding, and that `to_dict` / `from_dict` rebuild the same tokenizer.
