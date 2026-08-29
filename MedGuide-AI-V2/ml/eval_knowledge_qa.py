"""Measures the knowledge-question route: given a real MedQuAD question, does
the pipeline route it correctly and answer from the matching MedQuAD row?

Run:  python -m ml.eval_knowledge_qa [--limit N]
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.query_router import is_knowledge_question  # noqa: E402
from workflow import run_medguide  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MEDQUAD = ROOT / "data" / "medquad.csv"
RESULTS = ROOT / "results" / "knowledge_qa_metrics.json"
SAMPLE_SIZE = 200
RANDOM_STATE = 42

PATIENT = {
    "age": 30,
    "gender": "Female",
    "medical_history": "None reported",
    "allergies": "None reported",
    "medications": "None reported",
    "location": "Chennai",
}


def load_questions():
    csv.field_size_limit(10 ** 9)
    with open(MEDQUAD, newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if (row.get("question") or "").strip()
            and (row.get("answer") or "").strip()
        ]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=SAMPLE_SIZE)
    args = parser.parse_args()

    rows = load_questions()
    random.Random(RANDOM_STATE).shuffle(rows)
    sample = rows[: args.limit]

    routed = 0
    exact_question_at_1 = 0
    exact_question_at_3 = 0
    topic_at_1 = 0
    grounded = 0
    latencies = []

    for row in sample:
        question = row["question"].strip()
        topic = (row.get("focus_area") or "").strip().lower()

        if is_knowledge_question(question):
            routed += 1

        started = time.perf_counter()
        state = run_medguide({**PATIENT, "symptoms": question})
        latencies.append(time.perf_counter() - started)

        hits = state.get("medquad_hits", [])
        questions = [(h["question"] or "").strip().lower() for h in hits]
        topics = [(h["focus_area"] or "").strip().lower() for h in hits]

        if questions and questions[0] == question.lower():
            exact_question_at_1 += 1
        if question.lower() in questions:
            exact_question_at_3 += 1
        if topics and topic and topics[0] == topic:
            topic_at_1 += 1

        # The offline answer quotes the retrieved passage, so overlap with the
        # gold answer shows the reply really came from the corpus row.
        gold = set((row["answer"] or "").lower().split())
        reply = set((state.get("final_response") or "").lower().split())
        if gold and len(gold & reply) / len(gold) >= 0.15:
            grounded += 1

    n = len(sample)
    metrics = {
        "sample_size": n,
        "random_state": RANDOM_STATE,
        "llm": False,
        "routed_as_knowledge_question": round(routed / n, 4),
        "exact_source_question_at_1": round(exact_question_at_1 / n, 4),
        "exact_source_question_at_3": round(exact_question_at_3 / n, 4),
        "topic_at_1": round(topic_at_1 / n, 4),
        "answer_grounded_in_source_row": round(grounded / n, 4),
        "mean_latency_seconds": round(sum(latencies) / n, 4),
    }

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(metrics, indent=2))

    print(f"[knowledge-qa] {n} MedQuAD questions replayed through the pipeline")
    for key, value in metrics.items():
        print(f"    {key}: {value}")
    print(f"metrics-> {RESULTS}")


if __name__ == "__main__":
    main()
