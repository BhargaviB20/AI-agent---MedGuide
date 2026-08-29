"""Loads the trained models and exposes them to the agents.

Both loaders are cached and fail soft: if a model file has not been trained
yet, the agents fall back to their rule-based behaviour instead of crashing.
"""

from functools import lru_cache
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
INTENT_MODEL = MODEL_DIR / "intent_classifier.joblib"
TRIAGE_MODEL = MODEL_DIR / "triage_classifier.joblib"

# Most urgent first. Used to merge the rule-based screen with the classifier.
URGENCY_ORDER = ["EMERGENCY", "HIGH", "MODERATE", "LOW"]


@lru_cache(maxsize=1)
def _load(path_str: str):
    path = Path(path_str)
    if not path.exists():
        print(f"[predictors] {path.name} not trained yet, using rule-based fallback.")
        return None
    try:
        import joblib

        return joblib.load(path)
    except Exception as e:
        print(f"[predictors] could not load {path.name}: {e}")
        return None


def predict_intent(text: str):
    """Returns (intent, confidence) or (None, 0.0) if the model is unavailable."""
    model = _load(str(INTENT_MODEL))
    if model is None or not (text or "").strip():
        return None, 0.0

    intent = str(model.predict([text])[0])
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba([text])[0]))
    return intent, confidence


def predict_triage(text: str):
    """Returns (level, confidence) or (None, 0.0) if the model is unavailable.

    Applies the tuned asymmetric EMERGENCY threshold saved at training time:
    missing an emergency is much costlier than escalating a milder case.
    """
    bundle = _load(str(TRIAGE_MODEL))
    if bundle is None or not (text or "").strip():
        return None, 0.0

    pipeline = bundle["pipeline"]
    threshold = bundle.get("emergency_threshold", 0.5)

    classes = list(pipeline.classes_)
    probabilities = pipeline.predict_proba([text])[0]
    scores = dict(zip(classes, probabilities))

    level = str(classes[int(probabilities.argmax())])
    confidence = float(max(probabilities))
    if scores.get("EMERGENCY", 0.0) >= threshold:
        level = "EMERGENCY"
        confidence = float(scores["EMERGENCY"])
    return level, confidence


def most_urgent(*levels):
    """Safety-first merge: returns the most urgent of the given levels."""
    known = [level for level in levels if level in URGENCY_ORDER]
    if not known:
        return None
    return min(known, key=URGENCY_ORDER.index)
