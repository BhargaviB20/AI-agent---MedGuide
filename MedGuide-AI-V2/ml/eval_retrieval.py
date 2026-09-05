"""Measures the MedQuAD retrieval stage (the knowledge-acquisition component).

Two protocols are reported, because the deployed index contains the question
text itself, which makes question-as-query retrieval optimistically easy:

  as_deployed  index = question + answer + focus area (what the app uses)
  answer_only  index = answer + focus area only, so the query has to match the
               answer text; this is the honest generalisation setting

Metrics: Recall@1, Recall@3, Recall@5, MRR@10 for exact document hits, plus
topic-level Recall@3 (retrieved focus area equals the gold focus area).

    python -m ml.eval_retrieval
"""

import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "medquad.csv"
RESULTS_FILE = ROOT / "results" / "retrieval_metrics.json"

SAMPLE_SIZE = 300
RANDOM_STATE = 42
K_VALUES = [1, 3, 5]
MRR_DEPTH = 10


def load_docs():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    docs = []
    for row in rows:
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not question or not answer:
            continue
        docs.append({
            "question": question,
            "answer": answer,
            "focus_area": (row.get("focus_area") or "").strip(),
        })
    return docs


def index_text(doc, include_question):
    if include_question:
        return f"{doc['question']} {doc['answer']} {doc['focus_area']}"
    return f"{doc['answer']} {doc['focus_area']}"


def evaluate(docs, query_ids, include_question):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    matrix = vectorizer.fit_transform(
        [index_text(doc, include_question) for doc in docs]
    )

    hits = {k: 0 for k in K_VALUES}
    topic_hits_at_3 = 0
    reciprocal_ranks = []
    latencies = []

    for gold_index in query_ids:
        query = docs[gold_index]["question"]

        start = time.perf_counter()
        scores = cosine_similarity(vectorizer.transform([query]), matrix).flatten()
        ranked = scores.argsort()[::-1][:MRR_DEPTH]
        latencies.append((time.perf_counter() - start) * 1000)

        ranked = list(ranked)
        for k in K_VALUES:
            if gold_index in ranked[:k]:
                hits[k] += 1

        gold_topic = docs[gold_index]["focus_area"].lower()
        if gold_topic and any(
            docs[i]["focus_area"].lower() == gold_topic for i in ranked[:3]
        ):
            topic_hits_at_3 += 1

        if gold_index in ranked:
            reciprocal_ranks.append(1.0 / (ranked.index(gold_index) + 1))
        else:
            reciprocal_ranks.append(0.0)

    total = len(query_ids)
    latencies.sort()
    return {
        **{f"recall_at_{k}": round(hits[k] / total, 4) for k in K_VALUES},
        "topic_recall_at_3": round(topic_hits_at_3 / total, 4),
        f"mrr_at_{MRR_DEPTH}": round(sum(reciprocal_ranks) / total, 4),
        "retrieval_latency_ms_mean": round(sum(latencies) / total, 2),
        "retrieval_latency_ms_p95": round(latencies[int(0.95 * total) - 1], 2),
    }


def main():
    docs = load_docs()
    print(f"[retrieval] {len(docs)} MedQuAD documents indexed")

    random.seed(RANDOM_STATE)
    query_ids = random.sample(range(len(docs)), min(SAMPLE_SIZE, len(docs)))

    results = {
        "corpus": "MedQuAD (data/medquad.csv)",
        "documents": len(docs),
        "queries": len(query_ids),
        "retriever": "TF-IDF (English stop words, 20k features) + cosine similarity",
        "random_state": RANDOM_STATE,
        "protocols": {},
    }

    for name, include_question in [("as_deployed", True), ("answer_only", False)]:
        print(f"[retrieval] protocol: {name}")
        metrics = evaluate(docs, query_ids, include_question)
        results["protocols"][name] = metrics
        for key, value in metrics.items():
            print(f"    {key}: {value}")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"metrics-> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
