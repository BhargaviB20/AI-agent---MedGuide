"""Evaluates operation selection: how well does the orchestrator choose which
operation to run for a query?

Four policies are compared on the identical held-out split produced by
ml/train_planner.py, plus Gemini as a planner on a subsample when an API key
is configured:

    fixed_pipeline   always TRIAGE_SYMPTOMS (the original single-path system)
    rule_policy      hand-written keyword policy
    trained_planner  the trained classifier alone
    deployed_policy  trained classifier + deterministic safety overrides
    llm_planner      Gemini asked to pick the operation (subsample)

Besides accuracy and macro F1 the script reports the two error rates that
matter clinically:

    unsafe_routing_rate     emergencies not routed to EMERGENCY_ESCALATE
    unsafe_advice_rate      prescription requests not routed to MEDICATION_SAFETY

Routing an emergency to TRIAGE_SYMPTOMS is not yet a patient-visible failure:
the triage chain screens for red flags again and can raise the same alarm one
stage later. The script therefore also reports `system_safety`, the share of
emergency queries the whole pipeline escalates, through either route.

Run:  python -m ml.eval_planner
"""

import json
import random
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score

from agents import planner
from agents.risk_agent import rule_based_risk_level
from agents.symptom_agent import symptom_agent
from ml.predictors import most_urgent, predict_triage
from ml.train_planner import LABELS, RESULTS_FILE as TRAIN_RESULTS, SPLIT_FILE, load_rows

ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "results" / "planner_eval.json"

LLM_SAMPLE_SIZE = 90
RANDOM_STATE = 42


def unsafe_rate(y_true, y_pred, label):
    total = [i for i, truth in enumerate(y_true) if truth == label]
    if not total:
        return None
    missed = sum(1 for i in total if y_pred[i] != label)
    return round(missed / len(total), 4)


def summarise(y_true, y_pred, seconds):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "unsafe_routing_rate": unsafe_rate(y_true, y_pred, planner.EMERGENCY_ESCALATE),
        "unsafe_advice_rate": unsafe_rate(y_true, y_pred, planner.MEDICATION_SAFETY),
        "mean_decision_seconds": round(seconds / max(len(y_true), 1), 5),
        "per_class": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


def triage_level(text):
    """Risk level the triage chain would reach, without any Gemini call."""
    state = symptom_agent({"patient": {"symptoms": text, "age": 30}, "agent_log": []})
    rule_level, _ = rule_based_risk_level(state["patient"], state["symptom_details"])
    model_level, _ = predict_triage(text)
    if model_level is None:
        return rule_level
    screen = "EMERGENCY" if rule_level == "EMERGENCY" else None
    return most_urgent(screen, model_level) or model_level


def system_safety(texts, labels):
    """How the pipeline as a whole handles the emergency queries: escalated by
    the planner, escalated later by triage, or missed entirely."""
    emergencies = [
        text for text, label in zip(texts, labels)
        if label == planner.EMERGENCY_ESCALATE
    ]
    if not emergencies:
        return None

    bypassed = 0
    caught_by_triage = 0
    missed = []
    for text in emergencies:
        operation = planner.plan(text)["operation"]
        if operation == planner.EMERGENCY_ESCALATE:
            bypassed += 1
        elif operation == planner.TRIAGE_SYMPTOMS and triage_level(text) == "EMERGENCY":
            caught_by_triage += 1
        else:
            missed.append(text)

    total = len(emergencies)
    return {
        "emergency_queries": total,
        "escalated_by_planner": bypassed,
        "escalated_by_triage_stage": caught_by_triage,
        "escalated_overall_rate": round((bypassed + caught_by_triage) / total, 4),
        "missed_rate": round(len(missed) / total, 4),
        "missed_examples": missed[:10],
    }


def run(policy, texts):
    start = time.perf_counter()
    predictions = [policy(text) for text in texts]
    return predictions, time.perf_counter() - start


def main():
    texts, labels, _ = load_rows()

    if not SPLIT_FILE.exists():
        raise SystemExit("Run `python -m ml.train_planner` first (test split missing).")

    test_index = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))["test_index"]
    x_test = [texts[i] for i in test_index]
    y_test = [labels[i] for i in test_index]

    policies = {
        "fixed_pipeline": lambda text: planner.TRIAGE_SYMPTOMS,
        "rule_policy": planner.rule_plan,
        "trained_planner": lambda text: planner.model_plan(text)[0],
        "deployed_policy": lambda text: planner.plan(text)["operation"],
    }

    report = {
        "operations": LABELS,
        "test_size": len(x_test),
        "split_source": "results/planner_test_split.json (grouped 80/20)",
        "policies": {},
    }

    for name, policy in policies.items():
        predictions, seconds = run(policy, x_test)
        report["policies"][name] = summarise(y_test, predictions, seconds)
        row = report["policies"][name]
        print(f"{name:<16} acc {row['accuracy']:.4f}  macroF1 {row['f1_macro']:.4f}  "
              f"unsafe routing {row['unsafe_routing_rate']}  "
              f"unsafe advice {row['unsafe_advice_rate']}")

    report["system_safety"] = system_safety(x_test, y_test)
    safety = report["system_safety"]
    print(
        "system_safety    emergencies escalated {:.4f} "
        "(planner {}, triage stage {}), missed {:.4f}".format(
            safety["escalated_overall_rate"],
            safety["escalated_by_planner"],
            safety["escalated_by_triage_stage"],
            safety["missed_rate"],
        )
    )

    # Gemini as planner, on a subsample to keep the API cost bounded.
    rng = random.Random(RANDOM_STATE)
    sample = rng.sample(range(len(x_test)), min(LLM_SAMPLE_SIZE, len(x_test)))
    sample_texts = [x_test[i] for i in sample]
    sample_labels = [y_test[i] for i in sample]

    start = time.perf_counter()
    llm_predictions = [planner.llm_plan(text) for text in sample_texts]
    seconds = time.perf_counter() - start

    answered = [i for i, value in enumerate(llm_predictions) if value]
    if answered:
        report["llm_planner"] = summarise(
            [sample_labels[i] for i in answered],
            [llm_predictions[i] for i in answered],
            seconds,
        )
        report["llm_planner"]["sample_size"] = len(sample_texts)
        report["llm_planner"]["answered"] = len(answered)
        print(f"{'llm_planner':<16} acc {report['llm_planner']['accuracy']:.4f}  "
              f"macroF1 {report['llm_planner']['f1_macro']:.4f}  "
              f"answered {len(answered)}/{len(sample_texts)}")
    else:
        report["llm_planner"] = {
            "skipped": "Gemini unavailable (no API key or all calls failed)",
            "sample_size": len(sample_texts),
        }
        print("llm_planner      skipped (Gemini unavailable)")

    if TRAIN_RESULTS.exists():
        report["training_metrics_file"] = str(TRAIN_RESULTS.relative_to(ROOT))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"metrics-> {OUT_FILE}")


if __name__ == "__main__":
    main()
