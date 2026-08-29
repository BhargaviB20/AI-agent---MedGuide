CARE_PATHWAY = {
    "EMERGENCY": (
        "Emergency department",
        "Immediately",
        "Seek emergency care now. Do not wait to see if symptoms improve.",
    ),
    "HIGH": (
        "Doctor (general physician) or urgent care",
        "Within 24 hours",
        "Book an in-person consultation soon and rest until then.",
    ),
    "MODERATE": (
        "General physician",
        "Within 1-2 days if not improving",
        "Home care plus a doctor visit if symptoms persist or worsen.",
    ),
    "LOW": (
        "Self-care, pharmacist advice if needed",
        "Monitor at home",
        "Home care and monitoring are usually enough at this stage.",
    ),
}


def recommendation_agent(state):
    level = state.get("risk_level", "MODERATE")
    professional, urgency, next_step = CARE_PATHWAY.get(level, CARE_PATHWAY["MODERATE"])

    state["recommendation"] = (
        f"Suggested care: {professional}\n"
        f"Urgency: {urgency}\n"
        f"Next step: {next_step}"
    )

    state["agent_log"].append("Recommendation Agent completed.")
    return state
