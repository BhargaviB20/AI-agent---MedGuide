"""Trains the operation-selection classifier used by the planner agent.

Data : data/plan_dataset.csv (six operation labels, see ml/plan_dataset.py)
Model: TF-IDF (word 1-2 grams + char 3-5 grams) + Logistic Regression
Split: grouped 80 / 20 over source groups (corpus topic, seed vignette,
       condition pair, complaint, off-topic seed), so no group is split across
       the train/test boundary

Run:  python -m ml.train_planner
"""

import csv
import json
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

from agents.planner import OPERATIONS
from ml import plan_dataset

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "plan_dataset.csv"
MODEL_FILE = ROOT / "models" / "planner_classifier.joblib"
RESULTS_FILE = ROOT / "results" / "planner_metrics.json"
SPLIT_FILE = ROOT / "results" / "planner_test_split.json"

LABELS = list(OPERATIONS)
RANDOM_STATE = 42


def load_rows():
    if not DATASET.exists():
        print(f"[planner] {DATASET.name} not found, building it ...")
        plan_dataset.build()

    with open(DATASET, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    return (
        [r["text"] for r in rows],
        [r["label"] for r in rows],
        [r["group"] for r in rows],
    )


def build_pipeline():
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2
        )),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced")),
    ])


def grouped_split(texts, labels, groups):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    return next(splitter.split(texts, labels, groups))


def score(y_true, y_pred):
    present = [label for label in LABELS if label in set(y_true) | set(y_pred)]
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(
            precision_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "recall_macro": round(
            recall_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "per_class": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": {
            "labels": present,
            "matrix": confusion_matrix(y_true, y_pred, labels=present).tolist(),
        },
    }


def main():
    texts, labels, groups = load_rows()
    print(f"[planner] {len(texts)} queries, {len(set(groups))} groups, "
          f"{len(set(labels))} operations")

    train_index, test_index = grouped_split(texts, labels, groups)
    x_train = [texts[i] for i in train_index]
    y_train = [labels[i] for i in train_index]
    x_test = [texts[i] for i in test_index]
    y_test = [labels[i] for i in test_index]

    pipeline = build_pipeline()
    start = time.perf_counter()
    pipeline.fit(x_train, y_train)
    train_seconds = time.perf_counter() - start

    predicted = pipeline.predict(x_test)
    metrics = {
        "model": "TF-IDF (word 1-2 + char 3-5) + Logistic Regression",
        "dataset": "data/plan_dataset.csv",
        "operations": LABELS,
        "train_size": len(x_train),
        "test_size": len(x_test),
        "train_groups": len(set(groups[i] for i in train_index)),
        "test_groups": len(set(groups[i] for i in test_index)),
        "train_seconds": round(train_seconds, 2),
        "split": "GroupShuffleSplit 80/20 over source groups",
        "trained_planner": score(y_test, predicted),
    }

    folds = GroupKFold(n_splits=5)
    cv_scores = cross_val_score(
        build_pipeline(), texts, labels, groups=groups, cv=folds, scoring="accuracy"
    )
    metrics["cv5_grouped_accuracy_mean"] = round(float(cv_scores.mean()), 4)
    metrics["cv5_grouped_accuracy_std"] = round(float(cv_scores.std()), 4)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # The evaluation script compares the baselines on exactly this test set.
    SPLIT_FILE.write_text(
        json.dumps({"test_index": [int(i) for i in test_index]}, indent=2),
        encoding="utf-8",
    )

    print(classification_report(y_test, predicted, zero_division=0))
    print(f"accuracy                 {metrics['trained_planner']['accuracy']:.4f}")
    print(f"macro F1                 {metrics['trained_planner']['f1_macro']:.4f}")
    print(f"grouped 5-fold accuracy  {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"model  -> {MODEL_FILE}")
    print(f"metrics-> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
