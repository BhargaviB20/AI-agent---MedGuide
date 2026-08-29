EMERGENCY_KEYWORDS = [
    "chest pain", "severe difficulty breathing", "can't breathe",
    "cannot breathe", "unconscious", "unresponsive", "severe bleeding",
    "heavy bleeding", "suicidal", "stroke", "one side numb",
    "slurred speech", "seizure", "blue lips", "not breathing",
    "coughing blood", "vomiting blood",
]


def detect_emergency(symptom_text: str) -> bool:
    """Simple, deterministic keyword screen. Runs BEFORE the LLM chain so an
    emergency is flagged even if the API is slow, down, or unavailable."""
    text = (symptom_text or "").lower()
    return any(keyword in text for keyword in EMERGENCY_KEYWORDS)
