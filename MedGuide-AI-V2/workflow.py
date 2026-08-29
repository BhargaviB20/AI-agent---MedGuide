import time

from agents.emergency_check import detect_emergency
from agents.final_agent import final_agent
from agents.hospital_agent import hospital_agent
from agents.medical_agent import medical_agent
from agents.profile_agent import profile_agent
from agents.recommendation_agent import recommendation_agent
from agents.risk_agent import risk_agent
from agents.symptom_agent import symptom_agent

# Only the Final Response Agent calls the LLM. Every other agent is rule based,
# so a normal consultation costs exactly one API call and stays fast.
AGENT_STEPS = [
    ("Patient Profile Agent", profile_agent),
    ("Symptom Analysis Agent", symptom_agent),
    ("Medical Knowledge Agent", medical_agent),
    ("Risk Assessment Agent", risk_agent),
    ("Recommendation Agent", recommendation_agent),
    ("Hospital Navigation Agent", hospital_agent),
    ("Final Response Agent", final_agent),
]


def run_medguide(patient, on_step=None):
    """Runs the multi-agent pipeline.

    on_step: optional callback(step_name, index, total) invoked right before
    each agent runs, so the UI can show live progress.
    """
    start_time = time.perf_counter()
    state = {"patient": patient, "agent_log": []}

    if detect_emergency(patient.get("symptoms", "")):
        state["profile"] = dict(patient)
        state["symptoms"] = patient["symptoms"]
        state["symptom_details"] = {"identified_symptoms": [], "duration_days": None}
        state["medical_context"] = "Skipped: emergency fast-path."
        state["risk_level"] = "EMERGENCY"
        state["risk"] = "Risk level: EMERGENCY. Keyword-based safety screen triggered."
        state["recommendation"] = (
            "Seek immediate emergency medical care now (emergency department or "
            "local emergency number). Do not wait for further analysis."
        )
        state["hospital_navigation"] = (
            f"Go to the nearest emergency room near {patient['location']}, "
            "or call local emergency services."
        )
        state["final_response"] = (
            "### EMERGENCY\n\n"
            "**What this looks like**\n\n"
            "The symptoms you described can be a sign of a medical emergency.\n\n"
            "**What you should do now**\n\n"
            f"{state['recommendation']}\n\n"
            "**Warning signs - get care immediately**\n\n"
            "Trouble breathing, chest pain, fainting, confusion, heavy bleeding, "
            "or weakness on one side of the body.\n\n"
            "*This is an automated safety screen, not a diagnosis. If in doubt, "
            "treat it as an emergency.*"
        )
        state["agent_log"] = ["Emergency Fast-Path triggered - full agent chain skipped for speed."]
        state["response_time_seconds"] = round(time.perf_counter() - start_time, 2)
        return state

    total = len(AGENT_STEPS)
    for i, (step_name, agent) in enumerate(AGENT_STEPS, start=1):
        if on_step:
            on_step(step_name, i, total)
        state = agent(state)

    state["response_time_seconds"] = round(time.perf_counter() - start_time, 2)
    return state
