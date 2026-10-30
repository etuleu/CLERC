# Initial local baselines

## Frozen corpus

- Source: CLERC revision `ef042f8ab436f78704f17faa0a866d1b2b862f6f`.
- Source document count: **1,842,422**.
- Mini-corpus count: **18,424** (`floor(source_count / 100)`).
- Mandatory gold documents: **199** across 200 queries (one target shared).
- Remaining **18,225** documents sampled uniformly from non-gold documents.
- Sampling seed: 1729. Dev query seed: 42. Held-out test query seed: 2025.
- Corpus SHA-256: `2b175ffdddfd99e1a7952342a2396996682448ae7230a51ce2e0d03fb9c8036d`.
- JSONL corpus size: approximately **251 MiB**.
- All 200 queries have their exact gold document IDs in the corpus.

## Dev-100 results

Raw indirect/single-removed CLERC queries, exact document-ID judgments, full
opinion word-unigram index, lowercase, English stopwords, no stemming.

| Baseline | R@1 | R@5 | R@10 | R@100 | R@1000 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 (k1=1.2, b=0.75) | 34% | 48% | **58%** | 80% | 94% | 0.4065 | 0.4470 |
| TF-IDF cosine (sublinear TF) | 26% | 42% | **56%** | 79% | 98% | 0.3382 | 0.3895 |

Initial measured execution: approximately **12 seconds total** to fit both
baselines and evaluate 100 queries each; around **13 ms per query** for sparse
scoring/top-1000 ranking. Peak RAM approximately **914 MiB**. CPU-only; no remote
calls. Vocabulary: **191,217** terms. scikit-learn 1.6.1.

Held-out test-100 has been constructed but **not evaluated**. Tune on dev-100,
then use held-out evaluation sparingly.

## Interpretation

These scores are **not comparable to full-corpus CLERC or Cetient database
scores**: the mini-corpus has 100x fewer candidates, guaranteed gold coverage,
and source opinions are present only if randomly sampled or independently gold.
The purpose is rapid, controlled comparison of local retrieval experiments.

The local baseline is not an exact reproduction of the paper's Pyserini BM25
configuration, passage indexing or full-corpus evaluation.

## Artifacts

`data/clerc/mini-1pct/` contains the corpus, dev/test topics with gold document IDs,
source topics/qrels, sampling manifest and `runs/` with metrics and top-1000
rankings per query. See [README.md](README.md) for reproduction commands and the
experiment backlog.
