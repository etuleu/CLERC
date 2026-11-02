# Experiment: BM25 candidates + DREX relevance decisions

## Hypothesis

BM25 has high candidate recall but imperfect top-10 precision. A relevance
classifier may promote gold authorities from deeper in its candidate list.
Start with top 100 BM25 documents on **dev-100 only**. Never inject gold documents
into a query's candidate list; a missed first-stage gold remains a miss.

## Model/API reference

Reviewed: <https://deepinfra.com/nace-ai/drex-max>.

- POST `https://api.deepinfra.com/v1/decisions`.
- Model: `nace-ai/drex-max` (hosted alias; provider may update it).
- State: arbitrary text/JSON; typed questions.
- `noul` question returns a probability of yes, not generated text.
- At review time, page lists up to 262,144 tokens of context and automatic
  passage selection for states above 8,192 tokens, up to 16 questions per state.
- At review time, displayed price is $0.12/M tokens under a temporary promotion;
  verify current pricing before a larger run. The script saves actual token usage.

## Protocol

1. Load frozen local BM25 rankings and verify their corpus checksum.
2. Select top 100 documents for each raw development query.
3. Split each candidate into 250-word windows, stride 200.
4. Select two highest-scoring query-matching windows using BM25-style scoring
   with full mini-corpus IDF; send them in document order.
5. Send the original query and selected evidence to DREX, asking whether this
   candidate supplies the legal precedent missing at REDACTED.
6. Sort by probability of yes, break ties using original BM25 rank, retain top 10.
7. Evaluate exact CLERC document-ID R@10 against BM25 on identical queries.

This is pointwise relevance scoring, not a single listwise 'pick ten' request.
That avoids fitting 100 full opinions into one prompt and makes scores/cache
entries independently reusable. API requests contain no gold labels, candidate
IDs, original BM25 ranks, or corpus labels.

Query-focused excerpt selection and DREX are two simultaneous changes relative
to document BM25. Later ablations should compare full-document DREX, prefix
snippets, alternate relevance prompts, and conventional passage rerankers.

## Commands

```bash
# Prepare all dev-100 candidates offline. Does not require credentials.
.venv/bin/python benchmarks/mini_clerc/drex.py prepare

# Set a token in the environment (do not put it in tracked files).
export DEEPINFRA_TOKEN='...'

# Smoke run: first 5 dev queries x 100 candidates = 500 API calls.
.venv/bin/python benchmarks/mini_clerc/drex.py run --queries 5

# After checking schema, latency, usage and scores: full dev run (10,000 calls).
.venv/bin/python benchmarks/mini_clerc/drex.py run --queries 100

# Larger candidate pools must use a separate output directory.
.venv/bin/python benchmarks/mini_clerc/drex.py prepare --candidates 200 \
  --output data/clerc/mini-1pct/experiments/drex-bm25-200
```

Defaults: eight concurrent requests (`--workers 8`), no additional dependencies beyond the local
baseline environment. Only retry HTTP 429/transient server errors; never silently
replace failed model decisions with BM25. Network failures stop the run; rerun
to resume cached successes. After a successful smoke run, full execution reuses
its cached decisions. Caches are keyed by request content, including the prompt,
model, query and passages. Changing the prompt/model requires a new preparation.

Output: `data/clerc/mini-1pct/experiments/drex-bm25-100/`:

- `config.json`: model, exact instructions, corpus/candidate hashes, parameters,
  candidate-recall ceilings at several depths.
- `candidates.jsonl`: label-free raw queries and candidate excerpts.
- `cache/`: successful API responses, including probabilities and token usage.
- `rankings.jsonl`, `metrics.json`: scored rankings and metrics for completed run.

A partial failure leaves successful responses cached, but does not publish an
aggregate score treating failed candidates as irrelevant. Do not compare a
5-query smoke score to the full 100-query baseline.

## Status

Completed a five-query smoke test, then full dev-100 using cached smoke results.
No held-out test queries were evaluated. All 10,000 candidate decisions succeeded.

| Method | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 34% | 48% | 58% | 0.4065 | 0.4470 |
| BM25 top-100 + DREX | 48% | 73% | **77%** | 0.5808 | 0.6272 |

- R@10 gain: **19 percentage points**, 20 gained hits and 1 lost hit.
- Candidate-recall ceiling: 80%; DREX recovers 77 of 80 retrievable gold cases.
- Remaining misses: 20 outside top-100 pool; 3 in pool but below DREX rank 10.
- Smoke run: 65.6s. Full run reusing smoke cache: 1,068.2s. Combined live wall
  time approximately 18.9 minutes at four concurrent requests.
- Total usage across all decisions (including cached smoke decisions):
  14,695,171 input tokens; 354,747 output tokens. This is measured usage, not
  an invoice; verify provider billing separately.
- Scores above depth 100 cannot exceed 80%, because this method only ranks its
  100 input candidates; its `recall@1000` is not a full-corpus retrieval measure.

Artifacts are in the experiment output directory, including `comparison.json`
listing gained/lost queries and candidate-pool failures. API responses and raw
passages remain local, ignored by Git. Credentials are environment-only and are
never included in these artifacts.

## Candidate-depth experiment: up to 1,000

Completed dev-100 with the same prompt, passage selection and BM25 rankings,
using eight workers. Reused the 10,000 top-100 cached responses and scored
90,000 additional candidates. Evaluating subsets of the scored pool isolates
candidate depth without changing model decisions for shared candidates.

| BM25 depth | Candidate recall ceiling | DREX R@1 | DREX R@5 | **DREX R@10** |
|---|---:|---:|---:|---:|
| 100 | 80% | 48% | 73% | **77%** |
| 200 | 83% | 45% | 74% | **79%** |
| 500 | 86% | 45% | 74% | **79%** |
| 1,000 | 94% | 48% | 80% | **86%** |

- Top-1,000 improves R@10 by **9 percentage points** over top-100 and **28
  percentage points** over BM25 alone (58%).
- Relative to top-100 DREX: **11 gained hits, 2 lost hits**.
- Remaining misses: **6 outside the candidate pool**, **8 inside but below rank 10**.
- Top-1,000 MRR@10: **0.6137**; nDCG@10: **0.6740**.
- All 100,000 decisions succeeded eventually. A temporary DNS failure interrupted
  the initial run after 23 queries; successful calls were cached. The resumed
  run took **3,914.7 seconds** (~65 minutes), excluding the earlier interrupted
  segment and preparation. Do not interpret this as total fresh-run runtime.
- Total token usage for all cached/scored decisions: **146,476,751 input tokens**,
  **3,610,674 output tokens**. These totals include the reused top-100 responses;
  they are not incremental usage or a billing statement.
- Held-out test-100 remains **unevaluated**. All reported results are development
  results on the same gold-conditioned 1% corpus, not full-corpus CLERC results.

Artifacts: `data/clerc/mini-1pct/experiments/drex-bm25-1000/` contains
`config.json`, `candidates.jsonl`, `cache/`, `rankings.jsonl`, `metrics.json`,
`depth-comparison.json`, and `run.log`. The runner automatically produces
metrics at each available candidate depth. Raw artifacts remain ignored by Git.
