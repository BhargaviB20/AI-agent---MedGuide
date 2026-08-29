FACILITY = {
    "EMERGENCY": "nearest hospital emergency department",
    "HIGH": "nearby clinic or hospital outpatient department",
    "MODERATE": "local general physician clinic",
    "LOW": "local pharmacy or general physician clinic if needed",
}


def hospital_agent(state):
    level = state.get("risk_level", "MODERATE")
    location = state["patient"]["location"]

    state["hospital_navigation"] = (
        f"Look for a {FACILITY.get(level, FACILITY['MODERATE'])} near {location}. "
        "Real-time hospital availability is not connected in this prototype."
    )

    state["agent_log"].append("Hospital Navigation Agent completed.")
    return state
