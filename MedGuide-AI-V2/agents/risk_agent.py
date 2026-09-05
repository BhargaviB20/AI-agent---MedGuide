from agents.emergency_check import detect_emergency
from ml.predictors import most_urgent, predict_triage

HIGH_RISK_SYMPTOMS = [
    "breathing difficulty", "shortness of breath", "wheezing", "chest pain",
    "blood", "fainting", "confusion", "dehydration", "persistent vomiting",
]


def rule_based_risk_level(patient, details):
    """Deterministic risk scoring. Returns (level, reason).

    Runs without the API and without the trained model, so the safety screen
    works even when both are unavailable.
    """
    text = str(patient.get("symptoms") or "").lower()
    age = int(patient.get("age") or 0)
    duration = details.get("duration_days")
    severity = details.get("severity_indicators") or []

    if detect_emergency(text):
        return "EMERGENCY", "Emergency warning symptoms were mentioned."
    if any(s in text for s in HIGH_RISK_SYMPTOMS) or age >= 65 or age <= 2:
        return "HIGH", "Higher-risk symptoms or a more vulnerable age group."
    if (duration is not None and duration >= 7) or severity:
        return "MODERATE", "Symptoms have lasted a while or are described as severe."
    if (duration is not None and duration >= 3) or len(details.get("identified_symptoms", [])) >= 3:
        return "MODERATE", "Several symptoms together over a few days."
    return "LOW", "Mild, short-duration symptoms."


def risk_agent(state):
    """Hybrid triage.

    The trained classifier assigns the level; the deterministic red-flag
    screen can always escalate it to EMERGENCY but never lower it. If the
    model has not been trained, the rule ladder is used instead.
    """
    details = state.get("symptom_details", {})
    patient = state["patient"]

    rule_level, reason = rule_based_risk_level(patient, details)
    model_level, confidence = predict_triage(patient.get("symptoms"))

    if model_level is None:
        level = rule_level
    else:
        screen_level = "EMERGENCY" if rule_level == "EMERGENCY" else None
        level = most_urgent(screen_level, model_level) or model_level
        if level == "EMERGENCY" and screen_level:
            reason = "Emergency warning symptoms were mentioned."
        else:
            reason = (
                f"Trained triage classifier assigned {level} "
                f"(confidence {confidence:.2f})."
            )

    state["risk_level"] = level
    state["risk"] = f"Risk level: {level}. {reason}"
    state["risk_detail"] = {
        "rule_based_level": rule_level,
        "model_level": model_level,
        "model_confidence": round(confidence, 4),
        "final_level": level,
    }

    state["agent_log"].append("Risk Assessment Agent completed.")
    return state
