from rag.meddialog_retriever import (
    SYMPTOM_KEYWORDS,
    extract_duration_days,
    extract_keywords,
)

SEVERITY_WORDS = [
    "severe", "unbearable", "worst", "very bad", "high fever", "not improving",
    "getting worse", "worsening", "cannot", "can't", "unable",
]


def symptom_agent(state):
    """Rule-based extraction. No LLM call here, so the app stays fast and the
    single LLM call is spent on the final patient-facing answer."""
    text = state["patient"]["symptoms"]

    identified = [k for k in SYMPTOM_KEYWORDS if k in (text or "").lower()]
    if not identified:
        identified = extract_keywords(text)

    duration_days = extract_duration_days(text)
    severity = [w for w in SEVERITY_WORDS if w in (text or "").lower()]

    state["symptom_details"] = {
        "identified_symptoms": identified,
        "duration_days": duration_days,
        "severity_indicators": severity,
        "raw_text": text,
    }

    state["symptoms"] = (
        f"Reported symptoms: {', '.join(identified) if identified else text}\n"
        f"Duration: {duration_days if duration_days is not None else 'not stated'} day(s)\n"
        f"Severity words: {', '.join(severity) if severity else 'none'}"
    )

    state["agent_log"].append("Symptom Analysis Agent completed.")
    return state
