import json
import re
from functools import lru_cache
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "meddialog" / "english-train.json"

SYMPTOM_KEYWORDS = [
    "cold", "fever", "headache", "cough", "sore throat", "throat pain",
    "runny nose", "blocked nose", "stuffy nose", "sneezing", "body pain",
    "body ache", "fatigue", "tired", "weakness", "dizziness", "vomiting",
    "nausea", "diarrhea", "stomach pain", "abdominal pain", "acidity",
    "chest pain", "breathing difficulty", "shortness of breath", "wheezing",
    "phlegm", "mucus", "rash", "itching", "back pain", "joint pain",
    "ear pain", "eye pain", "burning urination", "constipation",
    "hives", "swelling", "acne", "pimples",
    "hair fall", "hairfall", "hair loss", "baldness", "dandruff",
    "toothache", "tooth pain", "bleeding gums", "mouth ulcer",
    "blurred vision", "red eye", "watery eyes",
    "insomnia", "anxiety", "stress", "low mood", "panic",
]

STOP_WORDS = {
    "i", "have", "having", "a", "an", "the", "for", "and", "or", "with",
    "my", "me", "is", "am", "are", "was", "been", "since", "what", "should",
    "do", "it", "to", "of", "from", "that", "this", "there", "also", "but",
    "in", "on", "at", "so", "very", "get", "got", "feel", "feeling", "days",
    "day", "week", "weeks", "month", "months", "please", "help", "doctor",
}


def extract_keywords(text):
    """Returns symptom phrases + duration mentions found in the user's text.
    Falls back to plain content words so retrieval never comes back empty."""
    text = (text or "").lower()

    found = [keyword for keyword in SYMPTOM_KEYWORDS if keyword in text]

    duration = re.findall(r"\b\d+\s*(?:day|days|week|weeks|month|months)\b", text)
    found.extend(duration)

    if not found:
        found = [w for w in re.findall(r"[a-z]{4,}", text) if w not in STOP_WORDS][:8]

    return found


def extract_duration_days(text):
    """Best-effort duration in days, used for risk scoring. None if unknown."""
    text = (text or "").lower()

    match = re.search(r"\b(\d+)\s*(day|days|week|weeks|month|months)\b", text)
    if not match:
        return None

    value = int(match.group(1))
    unit = match.group(2)

    if unit.startswith("week"):
        return value * 7
    if unit.startswith("month"):
        return value * 30
    return value


@lru_cache(maxsize=1)
def load_dataset():
    if not DATASET_PATH.exists():
        print(f"[meddialog] Dataset not found at {DATASET_PATH}")
        return []

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} MedDialog records.")
    return data


def retrieve_meddialog(symptoms, top_k=3):
    data = load_dataset()
    if not data:
        return []

    keywords = extract_keywords(symptoms)
    if not keywords:
        return []

    results = []

    for item in data:
        if isinstance(item, dict):
            description = str(item.get("description", ""))
            utterances = [str(u) for u in item.get("utterances", [])]
        else:
            description = str(item)
            utterances = []

        text = (description + " " + " ".join(utterances)).lower()

        score = sum(1 for keyword in keywords if keyword in text)
        if score == 0:
            continue

        doctor_answer = ""
        for utterance in utterances:
            if utterance.lower().startswith("doctor:"):
                doctor_answer = utterance[7:].strip()
                break

        results.append({
            "score": score,
            "patient": description.replace("patient:", "").strip()[:600],
            "doctor": doctor_answer[:800],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
