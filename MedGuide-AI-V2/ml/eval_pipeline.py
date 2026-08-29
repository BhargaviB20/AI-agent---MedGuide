"""End-to-end evaluation of the seven-agent pipeline.

Runs `run_medguide` over the held-out triage vignettes (the same
GroupShuffleSplit used by `ml.train_triage`, so no test vignette was seen in
training) and measures:

  triage accuracy        final risk level vs the labelled level
  CSVR                   critical safety violation rate: EMERGENCY cases that
                         did not end up as EMERGENCY
  escalation advice      EMERGENCY cases whose answer tells the patient to get
                         immediate care
  structure compliance   answers containing all five required sections
  groundedness           share of answer sentences whose clinical terms appear
                         in the retrieved MedQuAD/MedDialog context
  latency                per-agent and total wall-clock time

By default the Gemini call is skipped (offline fallback) so the run is free and
reproducible. Use --llm to evaluate the real generated answers instead:

    python -m ml.eval_pipeline
    python -m ml.eval_pipeline --llm --limit 8
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import GroupShuffleSplit

import workflow
from ml.train_triage import RANDOM_STATE

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "triage_dataset.csv"
RESULTS_FILE = ROOT / "results" / "pipeline_metrics.json"

REQUIRED_SECTIONS = [
    "What this looks like",
    "What you should do now",
    "Why this may have happened",
    "What you can do at home",
    "When you should see a doctor",
    "Warning signs",
]
IMMEDIATE_CARE_PHRASES = [
    "emergency", "immediately", "right now", "call an ambulance",
    "emergency room", "emergency department",
]
STOPWORDS = set("""a an and are as at be been but by can could do does for from
had has have how i if in is it its may me my not of on or our should so than
that the their them then there these they this to up us was we were what when
where which who will with you your yourself about after also any because
before being both each few more most other over same some such only own very
s t just don now""".split())


def load_holdout():
    with open(DATASET, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]
    groups = [int(r["seed_id"]) for r in rows]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    _, test_idx = next(splitter.split(texts, labels, groups))

    # One phrasing per held-out seed keeps the evaluation set free of duplicates.
    seen = set()
    cases = []
    for i in test_idx:
        if groups[i] in seen:
            continue
        seen.add(groups[i])
        cases.append({"symptoms": texts[i], "expected": labels[i], "seed_id": groups[i]})
    return cases


def content_words(text):
    words = re.findall(r"[a-z]{4,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


def groundedness(answer, context):
    """Share of answer sentences sharing at least two content words with the
    retrieved context. A coarse, fully reproducible proxy for grounding."""
    context_words = content_words(context or "")
    if not context_words:
        return None

    sentences = [s for s in re.split(r"[.\n]", answer) if len(s.split()) >= 5]
    if not sentences:
        return None

    grounded = sum(
        1 for s in sentences if len(content_words(s) & context_words) >= 2
    )
    return grounded / len(sentences)


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * len(ordered)) - 1))
    return round(ordered[index], 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="call Gemini for real answers")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only N cases")
    args = parser.parse_args()

    if not args.llm:
        # Force the deterministic offline fallback so the run costs nothing.
        import agents.final_agent as final_agent_module

        final_agent_module.generate = lambda prompt: ""

    cases = load_holdout()
    if args.limit:
        cases = cases[: args.limit]
    print(f"[pipeline] {len(cases)} held-out cases, llm={'on' if args.llm else 'off'}")

    agent_times = defaultdict(list)
    totals = []
    correct = 0
    emergency_total = 0
    emergency_missed = 0
    emergency_advice = 0
    structured = 0
    grounded_scores = []
    ai_used = 0
    records = []

    for case in cases:
        patient = {
            "age": 30,
            "gender": "not specified",
            "location": "Chennai",
            "medical_history": "none",
            "allergies": "none",
            "medications": "none",
            "symptoms": case["symptoms"],
        }

        step_start = {"t": time.perf_counter(), "name": None}

        def on_step(name, index, total):
            now = time.perf_counter()
            if step_start["name"]:
                agent_times[step_start["name"]].append(now - step_start["t"])
            step_start["name"] = name
            step_start["t"] = now

        state = workflow.run_medguide(patient, on_step=on_step)
        if step_start["name"]:
            agent_times[step_start["name"]].append(time.perf_counter() - step_start["t"])

        predicted = state["risk_level"]
        answer = state.get("final_response", "")
        totals.append(state["response_time_seconds"])
        correct += int(predicted == case["expected"])
        ai_used += int(bool(state.get("ai_used")))

        if case["expected"] == "EMERGENCY":
            emergency_total += 1
            emergency_missed += int(predicted != "EMERGENCY")
            lowered = answer.lower()
            emergency_advice += int(any(p in lowered for p in IMMEDIATE_CARE_PHRASES))

        present = sum(1 for s in REQUIRED_SECTIONS if s.lower() in answer.lower())
        structured += int(present >= 4)

        score = groundedness(answer, state.get("medical_context"))
        if score is not None:
            grounded_scores.append(score)

        records.append({
            "symptoms": case["symptoms"],
            "expected": case["expected"],
            "predicted": predicted,
            "seconds": state["response_time_seconds"],
        })

    total_cases = len(cases)
    metrics = {
        "cases": total_cases,
        "llm_enabled": args.llm,
        "test_set": "held-out seeds of data/triage_dataset.csv (GroupShuffleSplit, seed 42)",
        "triage_accuracy": round(correct / total_cases, 4),
        "emergency_cases": emergency_total,
        "critical_safety_violation_rate": round(emergency_missed / emergency_total, 4)
        if emergency_total else None,
        "immediate_care_advice_rate": round(emergency_advice / emergency_total, 4)
        if emergency_total else None,
        "structure_compliance": round(structured / total_cases, 4),
        "groundedness_mean": round(sum(grounded_scores) / len(grounded_scores), 4)
        if grounded_scores else None,
        "gemini_answer_rate": round(ai_used / total_cases, 4),
        "latency_seconds": {
            "mean": round(sum(totals) / total_cases, 3),
            "p50": percentile(totals, 0.5),
            "p95": percentile(totals, 0.95),
            "max": round(max(totals), 3),
        },
        "agent_latency_seconds_mean": {
            name: round(sum(values) / len(values), 4)
            for name, values in agent_times.items()
        },
        "cases_detail": records,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    for key in [
        "triage_accuracy", "critical_safety_violation_rate",
        "immediate_care_advice_rate", "structure_compliance",
        "groundedness_mean", "gemini_answer_rate",
    ]:
        print(f"    {key}: {metrics[key]}")
    print(f"    latency: {metrics['latency_seconds']}")
    print(f"    per-agent: {metrics['agent_latency_seconds_mean']}")
    print(f"metrics-> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
