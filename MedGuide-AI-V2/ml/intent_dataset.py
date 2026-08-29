"""Builds the query-intent dataset used to train the Master Orchestrator's router.

Labels are derived from the question templates that MedQuAD itself uses
(question column only, never the answer), so every label comes from the
dataset and not from hand annotation.
"""

import csv
import re
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "medquad.csv"
OUT_FILE = Path(__file__).resolve().parent.parent / "data" / "intent_dataset.csv"

# Ordered: the first pattern that matches wins.
INTENT_PATTERNS = [
    ("symptoms", r"^what are the symptoms of|^what are the signs"),
    ("treatment", r"^what are the treatments for|^what are the medications|^how is .* treated"),
    ("causes", r"^what causes|^what are the causes"),
    ("prevention", r"^how to prevent|^how can .* be prevented"),
    ("diagnosis", r"^how to diagnose|^how is .* diagnosed|^what tests"),
    ("genetics", r"^what are the genetic changes|^is .* inherited"),
    ("frequency", r"^how many people are affected"),
    ("prognosis", r"^what is the outlook for|^what are the stages of|^who is at risk for"),
    ("definition", r"^what is \(are\)|^do you have information about|^what is "),
]

CSV_FIELD_LIMIT = 10_000_000


def label_question(question: str):
    """Returns the intent label for a MedQuAD question, or None if unmatched."""
    q = (question or "").strip().lower()
    if not q:
        return None
    for label, pattern in INTENT_PATTERNS:
        if re.search(pattern, q):
            return label
    return None


def clean_question(question: str, focus_area: str) -> str:
    """Removes the disease name so the model learns the intent, not the disease.

    Without this the classifier could memorise topic words instead of the
    question type, which would inflate accuracy.
    """
    q = (question or "").strip()
    focus = (focus_area or "").strip()
    if focus and focus.lower() in q.lower():
        pattern = re.compile(re.escape(focus), re.IGNORECASE)
        q = pattern.sub("this condition", q)
    return re.sub(r"\s+", " ", q).strip()


def build(limit_per_class: int = 900):
    csv.field_size_limit(CSV_FIELD_LIMIT)
    rows = []
    counts = {}

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = label_question(row.get("question", ""))
            if label is None:
                continue
            if counts.get(label, 0) >= limit_per_class:
                continue
            text = clean_question(row.get("question", ""), row.get("focus_area", ""))
            if not text:
                continue
            rows.append({"text": text, "label": label})
            counts[label] = counts.get(label, 0) + 1

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    return rows, counts


if __name__ == "__main__":
    rows, counts = build()
    print(f"Wrote {len(rows)} labelled questions to {OUT_FILE}")
    for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label:12s} {count}")
