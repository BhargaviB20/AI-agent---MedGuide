"""Builds the operation-selection dataset used to train and evaluate the
planner (the orchestrator's decision step).

Each row is a user query labelled with the operation MedGuide should run:

    EMERGENCY_ESCALATE  red-flag presentation, skip everything else
    TRIAGE_SYMPTOMS     the user's own symptoms, run the triage chain
    RETRIEVE_KNOWLEDGE  general information question, answer from the corpus
    COMPARE_CONDITIONS  two conditions or treatments have to be retrieved
    MEDICATION_SAFETY   a prescription/dose request, which must be refused
    OUT_OF_SCOPE        not a medical question

Where each label comes from:
    RETRIEVE_KNOWLEDGE  real MedQuAD questions (label follows from the corpus)
    TRIAGE_SYMPTOMS     non-emergency triage vignettes (ml/triage_dataset.py)
    EMERGENCY_ESCALATE  emergency triage vignettes
    COMPARE_CONDITIONS  templates instantiated with real MedQuAD focus areas
    MEDICATION_SAFETY   authored prescription requests
    OUT_OF_SCOPE        authored non-medical requests

Every row carries a group id (the source question topic, seed vignette or
template). The train/test split is done over groups, so no phrasing or topic
seen in training reappears in the test set.
"""

import csv
import random
import re
import sys
from pathlib import Path

from ml import triage_dataset

ROOT = Path(__file__).resolve().parent.parent
MEDQUAD = ROOT / "data" / "medquad.csv"
OUT_FILE = ROOT / "data" / "plan_dataset.csv"

RANDOM_STATE = 42
PER_CLASS_CAP = 640

COMPARISON_TEMPLATES = [
    "what is the difference between {a} and {b}",
    "how is {a} different from {b}",
    "compare the symptoms of {a} and {b}",
    "{a} versus {b}, which is more serious",
    "is {a} the same as {b}",
    "difference between {a} and {b} treatment",
    "how do doctors tell {a} apart from {b}",
    "{a} vs {b} causes",
]

MEDICATION_TEMPLATES = [
    "which tablet should i take for {a}",
    "what medicine is best for {a}",
    "what is the dosage of paracetamol for {a}",
    "can i take antibiotics for {a}",
    "suggest a medicine for {a}",
    "how many mg of ibuprofen for {a}",
    "should i take steroids for {a}",
    "what drug do doctors prescribe for {a}",
    "name of the tablet used in {a}",
    "can i take two painkillers together for {a}",
]

OUT_OF_SCOPE_SEEDS = [
    "what is the capital of france",
    "who won the world cup last year",
    "write me a poem about the rain",
    "what is the weather tomorrow in chennai",
    "how do i fix my laptop wifi",
    "write a python program to sort a list",
    "translate good morning into french",
    "what is the price of bitcoin today",
    "give me a recipe for chocolate cake",
    "book me a flight to delhi",
    "who is the prime minister of india",
    "suggest a movie to watch tonight",
    "how do i use excel pivot tables",
    "what is the score of the cricket match",
    "help me with my maths homework",
    "how do i reset my phone password",
    "which song is number one right now",
    "tell me a joke",
    "what time does the train to bangalore leave",
    "how much does an iphone cost",
    "explain the rules of football",
    "write an essay about climate change",
    "what is the exam timetable for this semester",
    "which hotel is cheapest in goa",
    "how do i learn javascript quickly",
    "what is the stock price of infosys",
    "who acted in the latest marvel movie",
    "convert 100 dollars to rupees",
]

# Phrasing wrappers for the authored classes, so the model cannot win by
# memorising one fixed sentence shape per class.
WRAPPERS = ["{}", "{}?", "hi, {}", "please tell me {}", "doctor, {}"]

CSV_FIELD_LIMIT = 10_000_000


def medquad_rows():
    csv.field_size_limit(CSV_FIELD_LIMIT)
    with open(MEDQUAD, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clean_topic(text):
    topic = re.sub(r"\s+", " ", (text or "")).strip()
    return topic if 2 < len(topic) < 40 else ""


def build():
    rng = random.Random(RANDOM_STATE)
    rows = []

    def add(text, label, group):
        rows.append({"group": group, "text": text.strip(), "label": label})

    corpus = medquad_rows()

    # RETRIEVE_KNOWLEDGE: real corpus questions, grouped by topic.
    questions = [
        (r["question"].strip(), clean_topic(r.get("focus_area")))
        for r in corpus
        if (r.get("question") or "").strip()
    ]
    rng.shuffle(questions)
    seen_topics = set()
    for question, topic in questions:
        if len([r for r in rows if r["label"] == "RETRIEVE_KNOWLEDGE"]) >= PER_CLASS_CAP:
            break
        group = f"topic:{topic or question[:30]}"
        if group in seen_topics:
            continue
        seen_topics.add(group)
        add(question.lower(), "RETRIEVE_KNOWLEDGE", group)

    # TRIAGE_SYMPTOMS / EMERGENCY_ESCALATE: the triage vignettes.
    for seed_id, (text, urgency) in enumerate(triage_dataset.SEEDS):
        label = "EMERGENCY_ESCALATE" if urgency == "EMERGENCY" else "TRIAGE_SYMPTOMS"
        for wrapper in triage_dataset.WRAPPERS:
            add(wrapper.format(text), label, f"vignette:{seed_id}")

    # COMPARE_CONDITIONS: real disease names in comparison templates.
    topics = sorted({clean_topic(r.get("focus_area")) for r in corpus} - {""})
    rng.shuffle(topics)
    pairs = list(zip(topics[0::2], topics[1::2]))
    for index, (a, b) in enumerate(pairs):
        if len([r for r in rows if r["label"] == "COMPARE_CONDITIONS"]) >= PER_CLASS_CAP:
            break
        for template in COMPARISON_TEMPLATES:
            add(
                template.format(a=a.lower(), b=b.lower()),
                "COMPARE_CONDITIONS",
                f"pair:{index}",
            )

    # MEDICATION_SAFETY: prescription requests about real complaints.
    complaints = [
        "fever", "cold and cough", "headache", "body pain", "loose motion",
        "throat infection", "stomach pain", "skin allergy", "toothache",
        "back pain", "high blood pressure", "diabetes", "asthma",
        "urine infection", "period pain", "acidity", "migraine", "sinus",
        "ear pain", "eye redness", "anxiety", "sleeplessness", "vomiting",
        "chikungunya", "typhoid", "piles", "fungal infection", "sprained ankle",
    ]
    for index, complaint in enumerate(complaints):
        for template in MEDICATION_TEMPLATES:
            for wrapper in WRAPPERS[:3]:
                add(
                    wrapper.format(template.format(a=complaint)),
                    "MEDICATION_SAFETY",
                    f"complaint:{index}",
                )

    # OUT_OF_SCOPE: non-medical requests.
    for index, seed in enumerate(OUT_OF_SCOPE_SEEDS):
        for wrapper in WRAPPERS:
            add(wrapper.format(seed), "OUT_OF_SCOPE", f"offtopic:{index}")

    rows = cap_per_class(rows, PER_CLASS_CAP, rng)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    return rows


def cap_per_class(rows, cap, rng):
    """Keeps the classes comparable in size while keeping whole groups
    together, so capping cannot split a group across the train/test boundary."""
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], {}).setdefault(row["group"], []).append(row)

    kept = []
    for label, groups in by_label.items():
        names = sorted(groups)
        rng.shuffle(names)
        total = 0
        for name in names:
            if total >= cap:
                break
            kept.extend(groups[name])
            total += len(groups[name])
    return kept


if __name__ == "__main__":
    if not MEDQUAD.exists():
        sys.exit(f"{MEDQUAD} not found")

    built = build()
    counts = {}
    for row in built:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(f"Wrote {len(built)} queries to {OUT_FILE}")
    for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<20} {count}")
