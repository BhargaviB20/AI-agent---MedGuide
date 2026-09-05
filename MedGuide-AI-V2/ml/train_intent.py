"""Trains the query-intent router used by the Master Orchestrator agent.

Data : data/intent_dataset.csv (labels derived from MedQuAD question templates)
Model: TF-IDF (word 1-2 grams) + Logistic Regression
Split: stratified 80 / 20

Run:  python -m ml.train_intent
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
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from ml import intent_dataset

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "intent_dataset.csv"
MODEL_FILE = ROOT / "models" / "intent_classifier.joblib"
RESULTS_FILE = ROOT / "results" / "intent_metrics.json"

RANDOM_STATE = 42


def load_rows():
    if not DATASET.exists():
        print(f"[intent] {DATASET.name} not found, building it from medquad.csv ...")
        intent_dataset.build()

    with open(DATASET, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    return [r["text"] for r in rows], [r["label"] for r in rows]


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
    ])


def main():
    texts, labels = load_rows()
    print(f"[intent] {len(texts)} labelled questions, {len(set(labels))} classes")

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )

    pipeline = build_pipeline()
    start = time.perf_counter()
    pipeline.fit(x_train, y_train)
    train_seconds = time.perf_counter() - start

    predicted = pipeline.predict(x_test)

    accuracy = accuracy_score(y_test, predicted)
    macro_f1 = f1_score(y_test, predicted, average="macro")
    metrics = {
        "model": "TF-IDF (1-2 gram) + Logistic Regression",
        "dataset": "MedQuAD question templates (16,412 Q&A pairs)",
        "classes": sorted(set(labels)),
        "train_size": len(x_train),
        "test_size": len(x_test),
        "train_seconds": round(train_seconds, 2),
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision_score(y_test, predicted, average="macro"), 4),
        "recall_macro": round(recall_score(y_test, predicted, average="macro"), 4),
        "f1_macro": round(macro_f1, 4),
        "f1_weighted": round(f1_score(y_test, predicted, average="weighted"), 4),
        "per_class": classification_report(y_test, predicted, output_dict=True, zero_division=0),
        "confusion_matrix": {
            "labels": sorted(set(labels)),
            "matrix": confusion_matrix(y_test, predicted, labels=sorted(set(labels))).tolist(),
        },
    }

    cv_scores = cross_val_score(build_pipeline(), texts, labels, cv=5, scoring="accuracy")
    metrics["cv5_accuracy_mean"] = round(float(cv_scores.mean()), 4)
    metrics["cv5_accuracy_std"] = round(float(cv_scores.std()), 4)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(classification_report(y_test, predicted, zero_division=0))
    print(f"accuracy            {accuracy:.4f}")
    print(f"macro F1            {macro_f1:.4f}")
    print(f"5-fold CV accuracy  {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"model  -> {MODEL_FILE}")
    print(f"metrics-> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
