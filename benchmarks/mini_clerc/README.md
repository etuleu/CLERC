# CLERC mini-corpus: 1% of the released document collection

A small, offline experiment benchmark. Source data and generated runs live under
`data/clerc/mini-1pct/` (locally ignored), not in Git.

## Design

- Count the actual released `collection.doc.tsv.gz` records; retain `floor(N/100)`.
- Include **every gold document** for 100 development and 100 held-out test queries.
- Fill remaining slots with a uniform sample of non-gold documents (seed 1729).
- Dev queries: `random.Random(42).sample(topics, 100)` in release file order.
- Test queries: seed 2025, sampling 100 topics after excluding dev query IDs.
- Topics: official indirect, single-removed test queries. Raw query text is unchanged.
- Pin Hugging Face dataset revision `ef042f8ab436f78704f17faa0a866d1b2b862f6f`.
- Source opinions are **not explicitly included or excluded**; they can appear by
  random sampling. No post-retrieval source-opinion filtering.
- Ground truth is exact CLERC document-ID matching: no external API, inferred case
  labels, current CAP metadata, or name/citation heuristics are needed.

This is a **gold-conditioned mini-corpus**, not an unbiased full-corpus estimate.
It is deliberately easier than full CLERC: fewer distractors, guaranteed gold
coverage, and often no source opinion in the corpus. Do not compare its scores
numerically to the paper or Cetient's full-database scores as though corpus sizes
were the same. Always compare experiments against the same corpus checksum.

Both query sets originate in CLERC's published test release; the local `test`
label means a held-out subset for *our experiments*, not the original authors'
training/test split. Do not train on held-out query/gold pairs. Gold-document text
is legitimately searchable in the corpus, including the test gold documents.

## Run

From the repository root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r benchmarks/mini_clerc/requirements.txt

# One-time streaming scan of the existing 8.2 GB compressed collection.
.venv/bin/python benchmarks/mini_clerc/benchmark.py build

# Fit two local baselines and evaluate the development split.
.venv/bin/python benchmarks/mini_clerc/benchmark.py run

# Run only occasionally to validate improvements; keep test outputs uninspected.
.venv/bin/python benchmarks/mini_clerc/benchmark.py run --splits test

python -m unittest discover -s benchmarks/mini_clerc -v
```

`build` stores a corpus SHA-256 and exact source/sample counts in `manifest.json`.
It downloads only the small topics/qrels files, not the document collection.
For exact rebuilds, use the same source archive, dataset revision and Python RNG
implementation. Existing `build` output is overwritten; preserve a manifest before
changing sampling code or seeds.

## Baselines

Both are CPU-only sparse word-unigram baselines with lowercase tokenization,
scikit-learn English stopwords and **no stemming**. They index full opinion text.

- **BM25:** k1=1.2, b=0.75; Lucene-style nonnegative IDF; binary query term counts.
- **TF-IDF cosine:** sublinear term frequency, smoothed IDF, L2 normalization.

These are simple local reference implementations, not exact reproductions of the
paper's Pyserini BM25/passages setup. No GPU or network is needed for `run`.

Scores: macro Recall@1/5/10/100/1000, MRR@10 and binary-relevance nDCG@10.
Every query stays in the denominator. Duplicate document IDs are collapsed.
Zero-overlap documents are not arbitrarily ranked as matches. Equal scores use
stable corpus/document-ID order. Runs save metrics and scored top-1000 document
rankings for each query under `runs/`.

## Experiment backlog

Keep the corpus and raw-query baseline fixed; change one factor at a time.

1. BM25 tokenization, stemming and k1/b tuning (dev only).
2. Lexical + dense retrieval with reciprocal-rank fusion.
3. Passage chunking (300–500 words), MaxP aggregation to document IDs.
4. Rerank best-matching passages instead of opinion prefixes.
5. Query length, remaining-citation stripping, issue rewriting and multi-query fusion.
6. Legal-domain embeddings and released CLERC DPR models.
7. Candidate depths of 80, 100, 160, 500; measure candidate recall before reranking.
8. Fine-tune retriever/reranker on CLERC training triples only, with leakage checks.
9. Citation-graph expansion without access to hidden target labels.

Production Cetient follow-ups (separate corpus; report separately):

- Apply publication filtering before candidate truncation / refill filtered slots.
- Audit opinion-vs-case deduplication.
- Hybrid lexical/vector candidates and relevant-passage reranking.
- Record reranker success/fallback, latency and candidate counts explicitly.

Before drawing strong conclusions, audit old semantic-search-derived gold labels
and the rerank success/fallback path. The local mini-corpus evaluator avoids those
label-resolution problems by matching exact released document IDs.
