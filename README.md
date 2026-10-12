# RAG against the machine

A small Retrieval-Augmented Generation (RAG) system that answers questions about
the **vLLM 0.10.1** documentation and source code. It indexes `.py` and `.md`
files, retrieves the most relevant pieces with **BM25**, and generates a grounded
answer with **Qwen3-0.6B**.

## 1. System architecture

```
 data/raw/vllm-0.10.1/         raw .py and .md files
        |
        v
  [collect + read]   src/indexer.py     list files, read text
        |
        v
  [chunking]         src/chunker.py     pieces of at most 2000 characters
        |
        v
  [tokenizer]        src/tokenizer.py   same rule for index and query
        |
        v
  [inverted index]   src/indexer.py     data/processed/*.json
        |
        |  question
        v
  [BM25 retriever]   src/retriever.py   top-k (file, first char, last char)
        |
        v
  [context builder]  src/generator.py   re-reads the text from disk (12000 chars max)
        |
        v
  [Qwen3-0.6B]       src/generator.py   greedy answer, 256 new tokens max
```

| Component | File | Role |
|---|---|---|
| CLI | `src/cli.py`, `src/__main__.py` | Python Fire commands |
| Models | `src/models.py` | Pydantic models (questions, sources, results, answers) |
| Indexer | `src/indexer.py` | Build, save and load the index |
| Chunker | `src/chunker.py` | Split files into chunks |
| Tokenizer | `src/tokenizer.py` | Text to search tokens |
| Retriever | `src/retriever.py` | BM25 ranking |
| Evaluator | `src/evaluator.py` | IoU and Recall@k |
| Generator | `src/generator.py` | Context building and answer generation |

The retriever returns only **locations** (`file_path`, `first_character_index`,
`last_character_index`). The text itself is read back from the source files when
the context is built, so the answer always uses the real source text.

## 2. Chunking strategy

Every file is chunked on its own, so a chunk never mixes two files. A chunk is at
most **2000 characters**. Chunks are character ranges of the original file
(`first_character_index`, `last_character_index`) and the original line endings
are kept, so indexes match the source exactly.

* **Python files**: the file is parsed with `ast`. Cut points are placed at the
  start of every function and class (decorators included). Neighbouring pieces
  are then packed together while they stay under 2000 characters. If the file
  cannot be parsed (syntax error), it is cut line by line before packing.
* **Markdown files**: cut points are placed before each header (`#`), ignoring
  lines inside fenced code blocks. Pieces are packed up to 2000 characters.
* **Packing** (`pack`): small neighbouring pieces are merged until the next piece
  would exceed the limit. A single piece larger than the limit is cut at exactly
  2000 characters.

Result on vLLM 0.10.1: 1941 files, **13126 chunks**, none larger than 2000
characters.

## 3. Retrieval method

**Tokenizer.** Words are matched with `[A-Za-z0-9_]+` and lowercased. For an
identifier such as `get_cache_size`, the whole identifier **and** its parts
(`get`, `cache`, `size`) are indexed. This lets the query "cache size" match code
that only contains `get_cache_size`. The same function tokenizes documents and
queries, so both sides always agree.

**Inverted index.** For each token we store a list of `(chunk_id, tf)` pairs,
plus the length of each chunk. It is saved as three JSON files in
`data/processed/` (`chunks.json`, `postings.json`, `lengths.json`).

**Ranking: BM25** with `K1 = 1.5` and `B = 0.75`.

```
idf(w)   = log(1 + (N - n + 0.5) / (n + 0.5))
score    = sum over query words of  idf(w) * tf * (K1 + 1) / (tf + K1 * (1 - B + B * len / avg_len))
```

* `N` is the number of chunks and `n` the number of chunks containing the word.
* A rare word has a high `idf`, a word present everywhere has a low one.
* `tf` is saturated by `K1`, so repeating a word many times does not dominate.
* `B` normalises for chunk length, so long chunks are not favoured.

Only chunks that contain at least one query word are scored. The top `k` are
returned.

## 4. Performance analysis

Measured on the public datasets (`datasets_public`), with the vLLM 0.10.1 corpus
and the official moulinette (identical numbers to `evaluate`):

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Subject threshold |
|---|---|---|---|---|---|
| Docs (100 questions) | 60.0% | 75.0% | **82.0%** | 86.0% | Recall@5 >= 80% |
| Code (99 questions) | 41.4% | 54.5% | **60.6%** | 74.7% | Recall@5 >= 50% |

A source is considered found when it is in the same file and its IoU with the
ground truth is at least **0.05**.

| Measure | Result | Limit |
|---|---|---|
| Indexing (13126 chunks) | about 24 s | 300 s |
| Search, 200 questions (2 datasets) | about 25 s including index loading | 90 s |

Observations:
* Documentation questions are easier: the vocabulary of the question is usually
  the vocabulary of the page.
* Code questions are harder: questions use natural words while code uses
  identifiers. Splitting identifiers on `_` helps, but CamelCase names
  (`PagedAttention`) are kept as one token.
* Recall grows quickly with `k`, which shows that the right chunk is usually
  retrieved but not always ranked first.

## 5. Design decisions and trade-offs

* **BM25 instead of embeddings**: no GPU and no extra model, indexing takes
  seconds, results are easy to explain. The cost is that it matches words, not
  meaning ("configure" does not match "configuration").
* **No stemming**: simple and predictable, but word forms are not unified.
* **Whole identifier and its parts both indexed**: better recall on code, but
  the same text counts twice, which slightly inflates term frequency and chunk
  length. BM25 saturation limits the effect.
* **Chunks up to 2000 characters, cut on structure**: a function or a section
  stays together, which gives the generator useful context. Bigger chunks give
  more context but a lower IoU with small ground-truth ranges.
* **Retrieval returns positions, context is re-read from the files**: the
  retriever output is the exact `MinimalSource` format, and answers are grounded
  in the real source text.
* **JSON index**: human readable and safe to load (no pickle), a bit larger than
  a binary format.
* **Greedy decoding with a strict system prompt**: the model must answer only
  from the context, which reduces hallucination.
* **Known limitation**: a Markdown section larger than 2000 characters is cut at
  a fixed size, not at paragraph boundaries.

## 6. Example usage

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
make install        # uv sync
```

Put the vLLM sources in `data/raw/` (for example `data/raw/vllm-0.10.1/`), then:

```bash
# 1. Build the index (about 25 s) -> data/processed/
uv run python -m src index

# 2. Search one question
uv run python -m src search "How to configure OpenAI server?" --k 10

# 3. Generate a grounded answer for one question
uv run python -m src answer "How to configure OpenAI server?" --k 10

# 4. Search a whole dataset -> data/output/search_results/<dataset>.json
uv run python -m src search_dataset data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10

# 5. Evaluate Recall@1/3/5/10 against the ground truth
uv run python -m src evaluate \
    data/output/search_results/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json

# 6. Generate answers for a whole search result file
uv run python -m src answer_dataset data/output/search_results/dataset_docs_public.json
```

Other commands: `make lint`, `make lint-strict`, `make clean`.

### Supported model

Answer generation uses **Qwen/Qwen3-0.6B** (loaded in float32, about 2.4 GB of
RAM, downloaded automatically from Hugging Face on first use). Generation is
deterministic (greedy) and limited to 256 new tokens. The context passed to the
model is limited to 12000 characters.

### Errors handled

Empty query, `k` that is not a positive integer, missing or invalid dataset file,
missing index (a message asks to run `index` first), and unreadable files during
indexing (skipped).