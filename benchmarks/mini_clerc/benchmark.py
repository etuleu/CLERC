#!/usr/bin/env python3
"""Build and evaluate a deterministic, gold-conditioned 1% CLERC corpus."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import time
import urllib.request

REVISION = 'ef042f8ab436f78704f17faa0a866d1b2b862f6f'


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + '\n')


def build(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    for name in ['queries/test.single-removed.indirect.tsv', 'qrels/qrels-doc.test.indirect.tsv']:
        dest = root / Path(name).name
        if not dest.exists():
            urllib.request.urlretrieve(f'https://huggingface.co/datasets/jhu-clsp/CLERC/resolve/{REVISION}/{name}', dest)
    qrels = {}
    with (root / 'qrels-doc.test.indirect.tsv').open() as f:
        for qid, _, did, rel in csv.reader(f, delimiter='\t'):
            if int(rel) > 0:
                qrels.setdefault(qid, []).append(did)
    with (root / 'test.single-removed.indirect.tsv').open(newline='') as f:
        topics = [{'qid':r[0], 'query':r[1], 'gold_docids':qrels[r[0]]} for r in csv.reader(f, delimiter='\t')]
    dev = random.Random(42).sample(topics, 100)
    dev_ids = {x['qid'] for x in dev}
    test = random.Random(2025).sample([x for x in topics if x['qid'] not in dev_ids], 100)
    gold = {did for x in dev + test for did in x['gold_docids']}
    # Uniform reservoir of non-gold documents. 25k is safely above 1% of the
    # released corpus; trim uniformly after learning the exact source count.
    rng = random.Random(1729)
    reservoir, positives = [], {}
    count = non_gold = 0
    digest = hashlib.sha256()
    csv.field_size_limit(100_000_000)
    started = time.monotonic()
    with gzip.open(args.collection, 'rt', encoding='utf-8', newline='') as f:
        for row in csv.reader(f, delimiter='\t'):
            if len(row) != 2:
                raise ValueError(f'Expected docid/text at row {count + 1}')
            did, text = row
            count += 1
            if did in gold:
                positives[did] = text
            else:
                non_gold += 1
                if len(reservoir) < 25_000:
                    reservoir.append((did, text))
                else:
                    j = rng.randrange(non_gold)
                    if j < len(reservoir):
                        reservoir[j] = (did, text)
            if count % 200_000 == 0:
                print(f'Scanned {count:,} documents ({time.monotonic()-started:.0f}s)', flush=True)
    missing = gold - positives.keys()
    if missing:
        raise ValueError(f'Missing gold documents: {sorted(missing)}')
    target = count // 100
    if target < len(gold) or target - len(gold) > len(reservoir):
        raise ValueError('Reservoir capacity or gold-set size incompatible with target')
    selected = list(positives.items()) + rng.sample(reservoir, target - len(gold))
    selected.sort(key=lambda x: int(x[0]))
    with (root / 'corpus.jsonl').open('w') as f:
        for did, text in selected:
            line = json.dumps({'docid':did, 'text':text}, ensure_ascii=False) + '\n'
            f.write(line)
            digest.update(line.encode())
    save(root / 'dev.json', dev)
    save(root / 'test.json', test)
    save(root / 'manifest.json', {
        'dataset_revision': REVISION, 'source_documents':count,
        'corpus_documents':len(selected), 'fraction':len(selected)/count,
        'required_gold_documents':len(gold), 'sampling_seed':1729,
        'dev_seed':42, 'test_seed':2025, 'queries_per_split':100,
        'corpus_sha256':digest.hexdigest(),
        'sampling':'All dev/test qrel documents plus a uniform reservoir sample of other documents; floor(N/100) total.',
        'source_documents_not_excluded':True,
    })
    print((root / 'manifest.json').read_text())


def metrics(ranked, gold):
    """Macro recall; every evaluated query remains in the denominator."""
    recalls = {k: [] for k in (1, 5, 10, 100, 1000)}
    reciprocal, ndcg = [], []
    for qid, relevant in gold.items():
        relevant = set(relevant)
        docs = list(dict.fromkeys(ranked.get(qid, [])))
        for k in recalls:
            recalls[k].append(len(set(docs[:k]) & relevant) / len(relevant))
        ranks = [i for i,d in enumerate(docs, 1) if d in relevant]
        reciprocal.append(1/min(ranks) if ranks and min(ranks) <= 10 else 0)
        dcg = sum(1/math.log2(i+1) for i in ranks if i <= 10)
        ideal = sum(1/math.log2(i+1) for i in range(1, min(len(relevant),10)+1))
        ndcg.append(dcg/ideal)
    return {**{f'recall@{k}':sum(v)/len(v) for k,v in recalls.items()},
            'mrr@10':sum(reciprocal)/len(reciprocal), 'ndcg@10':sum(ndcg)/len(ndcg)}


def run(args):
    # Imported only for scoring so building the corpus uses stdlib alone.
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    import sklearn
    root = Path(args.output)
    manifest = json.loads((root / 'manifest.json').read_text())
    started = time.monotonic()
    corpus = [json.loads(line) for line in (root / 'corpus.jsonl').open()]
    docids = np.array([x['docid'] for x in corpus])
    vectorizer = CountVectorizer(stop_words='english', dtype=np.float32)
    counts = vectorizer.fit_transform(x['text'] for x in corpus)
    print(f'Indexed {len(corpus):,} docs, {counts.shape[1]:,} terms in {time.monotonic()-started:.1f}s', flush=True)
    del corpus
    # Lucene-style nonnegative IDF, BM25 k1=1.2 b=0.75. No stemming.
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log1p((len(docids)-df+0.5)/(df+0.5)).astype(np.float32)
    bm25 = counts.copy()
    norms = 1.2 * (0.25 + 0.75 * lengths / lengths.mean())
    # Row-wise sparse operations avoid a large repeated document-length array.
    for i in range(bm25.shape[0]):
        a,z = bm25.indptr[i:i+2]
        bm25.data[a:z] = bm25.data[a:z]*2.2/(bm25.data[a:z]+norms[i])
    bm25 = bm25.multiply(idf).tocsr()
    transformer = TfidfTransformer(sublinear_tf=True)
    tfidf = transformer.fit_transform(counts)
    for split in args.splits:
        topics = json.loads((root / f'{split}.json').read_text())
        qcounts = vectorizer.transform(x['query'] for x in topics)
        # BM25 uses binary query terms, avoiding repeated-term over-weighting.
        binary = qcounts.copy()
        binary.data[:] = 1
        queries = {'bm25':binary, 'tfidf':transformer.transform(qcounts)}
        for model, matrix in [('bm25', bm25), ('tfidf', tfidf)]:
            ranked, records, latencies = {}, [], []
            for i,t in enumerate(topics):
                tick = time.perf_counter()
                scores = (matrix @ queries[model][i].T).toarray().ravel()
                # Stable deterministic docid-order tie break. Do not rank zero matches.
                order = np.argsort(-scores, kind='stable')[:1000]
                order = order[scores[order] > 0]
                docs = docids[order].tolist()
                latency = (time.perf_counter()-tick)*1000
                latencies.append(latency)
                ranked[t['qid']] = docs
                records.append({'qid':t['qid'], 'docids':docs,
                                'scores':scores[order].tolist(), 'latency_ms':latency})
            result = metrics(ranked, {x['qid']:x['gold_docids'] for x in topics})
            result.update({'model':model, 'split':split, 'queries':len(topics),
                           'mean_query_ms':float(np.mean(latencies)),
                           'corpus_documents':len(docids),
                           'corpus_sha256':manifest['corpus_sha256'],
                           'sklearn_version':sklearn.__version__,
                           'tokenization':'sklearn word unigrams, lowercase, English stopwords, no stemming',
                           'parameters':{'k1':1.2,'b':0.75,'binary_query':True} if model=='bm25' else {'sublinear_tf':True,'norm':'l2'}})
            out = root / 'runs'
            out.mkdir(exist_ok=True)
            save(out / f'{split}-{model}-metrics.json',result)
            with (out / f'{split}-{model}-rankings.jsonl').open('w') as f:
                for r in records:
                    f.write(json.dumps(r)+'\n')
            print(json.dumps(result,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['build','run'])
    parser.add_argument('--collection',default='data/clerc/collection.doc.tsv.gz')
    parser.add_argument('--output',default='data/clerc/mini-1pct')
    parser.add_argument('--splits',nargs='+',choices=['dev','test'],default=['dev'])
    args = parser.parse_args()
    (build if args.task=='build' else run)(args)


if __name__ == '__main__':
    main()
