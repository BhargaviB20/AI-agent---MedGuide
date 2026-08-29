def profile_agent(state):
    p = state["patient"]
    state["profile"] = {
        "age": p["age"],
        "gender": p["gender"],
        "medical_history": p["medical_history"],
        "allergies": p["allergies"],
        "medications": p["medications"],
        "location": p["location"],
    }
    state["agent_log"].append("Patient Profile Agent completed.")
    return state
