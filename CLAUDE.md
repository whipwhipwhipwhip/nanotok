# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch reimplementation of BPE tokenization (nanoGPT-lecture style), built to
understand the algorithm rather than to replace `tiktoken`. `TiktokenBPETokenizer` runs our
own merge loop but is seeded from a real tiktoken encoding's learned ranks, so its output can
be checked against `tiktoken.encode` directly — that equivalence is the project's main
correctness signal (`tests/nanotok/test_bpe.py::test_matches_tiktoken`).

## Commands

```bash
uv sync                    # install deps (uses uv_build backend, Python >=3.12)
uv run pytest -q           # run the full test suite
uv run pytest tests/nanotok/test_char.py::test_round_trip   # single test
uv run pytest -k specials  # by keyword
```

There is no lint/format/typecheck config in this repo — don't invent commands for them.

## Architecture

Four tokenizers share one `Tokenizer` ABC (`src/nanotok/base.py`: `vocab_size`, `encode`,
`decode`, `to_dict`/`from_dict`), but split into two unrelated inheritance branches because
their id spaces aren't refinements of each other:

- **`CharTokenizer`** (`char.py`) — one id per distinct character. A sibling of the byte
  branch, not a subclass of it. Frozen dataclass; `from_text` derives the alphabet from a
  corpus, `from_alphabet` states it up front (use this for a task with a known closed
  character set, so ids don't shift when the corpus changes).
- **`ByteTokenizer`** (`byte.py`) — one id per byte, vocab 256, no training. Base of the BPE
  branch: a BPE tokenizer with zero merges *is* a `ByteTokenizer`, so subclasses inherit
  `decode`/`vocab_size` and override only `_encode_ordinary`. Also owns special-token
  handling (`special_map`, splitting specials out before merging in `encode`) since that
  mechanism is identical regardless of where the merge ids come from.
  - **`RegexTokenizer`** (`regex_bpe.py`) — trains its own merges from a corpus, splitting on
    a regex pattern first so merges never cross chunk boundaries (e.g. across whitespace).
    Special tokens are numbered directly above the 256 bytes, in declaration order.
  - **`TiktokenBPETokenizer`** (`tiktoken_bpe.py`) — merges recovered from a tiktoken
    encoding's `mergeable_ranks` (`recover_merges`/`bpe` re-run BPE per token to reconstruct
    *which* pair produced each rank, since `mergeable_ranks` only stores the finished byte
    string). Bytes are permuted into tiktoken's rank space before merging and back after
    decoding (`byte_shuffle`). Special tokens keep the encoding's own ids rather than being
    renumbered, so output stays interchangeable with `tiktoken.encode`. Not trainable —
    `train()` raises.

`merges.py` holds the byte-pair primitives shared by both BPE tokenizers (`merge`,
`find_largest`, `apply_merges`, `build_vocab`, special-token splitting/validation) — kept in
one place specifically so the trained and tiktoken-seeded tokenizers agree on what a merge is.

`src/nanotok/__init__.py` maps each concrete class's `kind` tag to itself (`_KINDS`) so
`from_dict` can reconstruct any tokenizer from its `to_dict()` output without the caller
naming a class, and exposes `fingerprint()` (sha256 of the JSON recipe) as a tokenizer's
identity for provenance/caching purposes.

### Design invariants worth preserving

- `to_dict()` is a *recipe* (e.g. an encoding name or an alphabet string), never a dump of
  learned state (never 50k merges or two lookup dicts) — it must stay JSON-serialisable and
  deterministic, since its hash is the tokenizer's fingerprint.
- Special tokens are always declared, never learned from a corpus — this is what keeps their
  ids a fact about the config rather than about training data.
- `RegexTokenizer` numbers its declared specials itself (above the 256 bytes);
  `TiktokenBPETokenizer` takes the ids the encoding already assigned. Don't unify these —
  they disagree for a reason (see `tiktoken_bpe.py` module docstring).
- `vocab_size` is always "one past the highest id", not a count of entries — cl100k's
  specials leave a gap at 100256 that still has to be paid for in an embedding table.
- `pad_vocab_to` (all four tokenizers, `base.pad_vocab_size`) rounds `vocab_size` up to a
  GPU-friendly number — e.g. `TiktokenBPETokenizer("gpt2", pad_vocab_to=50304)`. It must
  change `vocab_size` and nothing else: no id moves, `train`'s ceiling counts merges rather
  than padding, and the key is omitted from `to_dict` when unpadded so old fingerprints
  still match. It's absolute, not a multiple to round to — the caller does that arithmetic.

## Tests

`tests/nanotok/test_bpe.py` parametrizes over both `gpt2` and `cl100k_base` encodings and
over a fixed list of tricky `SAMPLES` (contractions, digit runs, multi-byte characters,
leading/trailing/repeated whitespace) chosen because they're where a naive BPE implementation
diverges from GPT-2/GPT-4's real tokenizers. When changing merge/encode logic, keep
`test_matches_tiktoken` passing — it's the check that the reimplementation is actually right,
not just self-consistent.
