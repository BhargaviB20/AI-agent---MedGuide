from agents.emergency_check import detect_emergency

HIGH_RISK_SYMPTOMS = [
    "breathing difficulty", "shortness of breath", "wheezing", "chest pain",
    "blood", "fainting", "confusion", "dehydration", "persistent vomiting",
]


def risk_agent(state):
    """Deterministic risk scoring. Fast, reproducible and easy to defend in a
    viva — the LLM is only used for the final explanation."""
    details = state.get("symptom_details", {})
    text = str(state["patient"]["symptoms"]).lower()
    age = int(state["patient"].get("age") or 0)
    duration = details.get("duration_days")
    severity = details.get("severity_indicators") or []

    if detect_emergency(text):
        level, reason = "EMERGENCY", "Emergency warning symptoms were mentioned."
    elif any(s in text for s in HIGH_RISK_SYMPTOMS) or age >= 65 or age <= 2:
        level, reason = "HIGH", "Higher-risk symptoms or a more vulnerable age group."
    elif (duration is not None and duration >= 7) or severity:
        level, reason = "MODERATE", "Symptoms have lasted a while or are described as severe."
    elif (duration is not None and duration >= 3) or len(details.get("identified_symptoms", [])) >= 3:
        level, reason = "MODERATE", "Several symptoms together over a few days."
    else:
        level, reason = "LOW", "Mild, short-duration symptoms."

    state["risk_level"] = level
    state["risk"] = f"Risk level: {level}. {reason}"

    state["agent_log"].append("Risk Assessment Agent completed.")
    return state
