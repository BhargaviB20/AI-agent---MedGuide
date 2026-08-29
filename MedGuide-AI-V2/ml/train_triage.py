"""Trains the triage-urgency classifier used by the Risk Assessment agent.

Data : data/triage_dataset.csv (ESI-style labels: EMERGENCY / HIGH / MODERATE / LOW)
Model: TF-IDF (word 1-2 grams + char 3-5 grams) + Linear SVM
Split: grouped 80 / 20 over seed vignettes, so no phrasing of a test vignette
       appears in training

The script also reports the same metrics for the existing rule-based
risk_agent on the identical test set, which is the single-agent style baseline
the paper compares against.

Run:  python -m ml.train_triage
"""

import csv
import json
import time
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from agents.emergency_check import detect_emergency
from agents.risk_agent import rule_based_risk_level
from agents.symptom_agent import symptom_agent
from ml import triage_dataset
from ml.predictors import most_urgent

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "triage_dataset.csv"
MODEL_FILE = ROOT / "models" / "triage_classifier.joblib"
RESULTS_FILE = ROOT / "results" / "triage_metrics.json"

LABELS = ["EMERGENCY", "HIGH", "MODERATE", "LOW"]
RANDOM_STATE = 42


def load_rows():
    if not DATASET.exists():
        print(f"[triage] {DATASET.name} not found, building it ...")
        triage_dataset.build()

    with open(DATASET, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]
    groups = [int(r["seed_id"]) for r in rows]
    return texts, labels, groups


def build_pipeline():
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ])
    return Pipeline([
        ("features", features),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced"), cv=3, method="sigmoid"
        )),
    ])


def predict_with_threshold(pipeline, texts, threshold):
    """Argmax prediction, escalated to EMERGENCY when P(EMERGENCY) >= threshold.

    Under-triage is far more harmful than over-triage, so the decision rule for
    the EMERGENCY class is deliberately asymmetric.
    """
    classes = list(pipeline.classes_)
    emergency_index = classes.index("EMERGENCY")
    probabilities = pipeline.predict_proba(texts)
    predictions = []
    for row in probabilities:
        label = classes[int(row.argmax())]
        if row[emergency_index] >= threshold:
            label = "EMERGENCY"
        predictions.append(label)
    return predictions


def tune_threshold(texts, labels, groups, max_undertriage=0.05):
    """Picks the largest threshold whose under-triage rate stays within budget.

    Tuned on a grouped validation split carved out of the training data only,
    so the test set is never used for model selection.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    inner_train, inner_val = next(splitter.split(texts, labels, groups))

    pipeline = build_pipeline()
    pipeline.fit([texts[i] for i in inner_train], [labels[i] for i in inner_train])

    x_val = [texts[i] for i in inner_val]
    y_val = [labels[i] for i in inner_val]

    best = 0.5
    for threshold in [round(0.02 * i, 2) for i in range(1, 26)]:
        predictions = predict_with_threshold(pipeline, x_val, threshold)
        if undertriage_rate(y_val, predictions) <= max_undertriage:
            best = threshold
    return best


def score(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "per_class": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        },
    }


def rule_based_predictions(texts):
    """Runs the full rule ladder so both systems see the same inputs."""
    predictions = []
    for text in texts:
        state = {"patient": {"age": 30, "symptoms": text}, "agent_log": []}
        state = symptom_agent(state)
        level, _ = rule_based_risk_level(state["patient"], state["symptom_details"])
        predictions.append(level)
    return predictions


def keyword_screen_predictions(texts):
    """Only the deterministic red-flag keyword screen, which is kept as the
    guardrail in the deployed pipeline."""
    return ["EMERGENCY" if detect_emergency(text.lower()) else None for text in texts]


def undertriage_rate(y_true, y_pred):
    """Fraction of EMERGENCY cases predicted as anything less urgent.

    This is the paper's Critical Safety Violation Rate for the triage stage.
    """
    total = sum(1 for y in y_true if y == "EMERGENCY")
    if not total:
        return 0.0
    missed = sum(1 for t, p in zip(y_true, y_pred) if t == "EMERGENCY" and p != "EMERGENCY")
    return round(missed / total, 4)


def main():
    texts, labels, groups = load_rows()
    print(f"[triage] {len(texts)} vignettes from {len(set(groups))} seeds")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(texts, labels, groups))

    x_train = [texts[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    x_test = [texts[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    pipeline = build_pipeline()
    start = time.perf_counter()
    pipeline.fit(x_train, y_train)
    train_seconds = time.perf_counter() - start

    threshold = tune_threshold(x_train, y_train, [groups[i] for i in train_idx])
    print(f"[triage] EMERGENCY escalation threshold tuned on validation split: {threshold}")

    predicted = predict_with_threshold(pipeline, x_test, threshold)
    argmax_predicted = list(pipeline.predict(x_test))
    rule_predicted = rule_based_predictions(x_test)
    # Deployed configuration: the classifier decides the level, but the
    # deterministic red-flag screen can always escalate it to EMERGENCY.
    screen_predicted = keyword_screen_predictions(x_test)
    hybrid_predicted = [most_urgent(screen, model) or model
                        for screen, model in zip(screen_predicted, predicted)]

    metrics = {
        "model": "TF-IDF (word 1-2 gram + char 3-5 gram) + Linear SVM",
        "dataset": "ESI-style labelled symptom vignettes",
        "labels": LABELS,
        "seeds": len(set(groups)),
        "train_size": len(x_train),
        "test_size": len(x_test),
        "split": "GroupShuffleSplit over seed vignettes (no phrasing overlap)",
        "train_seconds": round(train_seconds, 2),
        "emergency_threshold": threshold,
        "trained_model_argmax": score(y_test, argmax_predicted),
        "trained_model": score(y_test, predicted),
        "rule_based_baseline": score(y_test, rule_predicted),
        "hybrid_deployed": score(y_test, hybrid_predicted),
        "undertriage_rate_trained_argmax": undertriage_rate(y_test, argmax_predicted),
        "undertriage_rate_trained": undertriage_rate(y_test, predicted),
        "undertriage_rate_rule_based": undertriage_rate(y_test, rule_predicted),
        "undertriage_rate_hybrid": undertriage_rate(y_test, hybrid_predicted),
    }

    cv_scores = cross_val_score(
        build_pipeline(), texts, labels, groups=groups,
        cv=GroupKFold(n_splits=5), scoring="accuracy"
    )
    metrics["cv5_accuracy_mean"] = round(float(cv_scores.mean()), 4)
    metrics["cv5_accuracy_std"] = round(float(cv_scores.std()), 4)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "emergency_threshold": threshold, "labels": LABELS},
        MODEL_FILE,
    )
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n--- trained classifier ---")
    print(classification_report(y_test, predicted, zero_division=0))
    print("--- existing rule-based screen (baseline) ---")
    print(classification_report(y_test, rule_predicted, zero_division=0))
    print("--- hybrid, as deployed in risk_agent ---")
    print(classification_report(y_test, hybrid_predicted, zero_division=0))
    print(f"trained  accuracy {metrics['trained_model']['accuracy']:.4f} "
          f"macro F1 {metrics['trained_model']['f1_macro']:.4f} "
          f"under-triage {metrics['undertriage_rate_trained']:.4f}")
    print(f"baseline accuracy {metrics['rule_based_baseline']['accuracy']:.4f} "
          f"macro F1 {metrics['rule_based_baseline']['f1_macro']:.4f} "
          f"under-triage {metrics['undertriage_rate_rule_based']:.4f}")
    print(f"hybrid   accuracy {metrics['hybrid_deployed']['accuracy']:.4f} "
          f"macro F1 {metrics['hybrid_deployed']['f1_macro']:.4f} "
          f"under-triage {metrics['undertriage_rate_hybrid']:.4f}")
    print(f"5-fold CV accuracy {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"model  -> {MODEL_FILE}")
    print(f"metrics-> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
