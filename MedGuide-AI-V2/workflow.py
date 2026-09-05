import time

from agents.compare_agent import compare_agent
from agents.final_agent import final_agent
from agents.hospital_agent import hospital_agent
from agents.knowledge_agent import knowledge_agent
from agents.medical_agent import medical_agent
from agents.planner import (
    COMPARE_CONDITIONS,
    EMERGENCY_ESCALATE,
    MEDICATION_SAFETY,
    OUT_OF_SCOPE,
    RETRIEVE_KNOWLEDGE,
    TRIAGE_SYMPTOMS,
    plan,
)
from agents.profile_agent import profile_agent
from agents.recommendation_agent import recommendation_agent
from agents.risk_agent import risk_agent
from agents.safety_agent import medication_safety_agent, out_of_scope_agent
from agents.symptom_agent import symptom_agent

# The planner picks one of these operations per query; only the chosen chain
# runs. Every step except the Final Response Agent is rule based or a small
# local model, so a consultation costs at most one LLM call.
OPERATION_CHAINS = {
    TRIAGE_SYMPTOMS: [
        ("Patient Profile Agent", profile_agent),
        ("Symptom Analysis Agent", symptom_agent),
        ("Medical Knowledge Agent", medical_agent),
        ("Risk Assessment Agent", risk_agent),
        ("Recommendation Agent", recommendation_agent),
        ("Hospital Navigation Agent", hospital_agent),
        ("Final Response Agent", final_agent),
    ],
    RETRIEVE_KNOWLEDGE: [
        ("Patient Profile Agent", profile_agent),
        ("Symptom Analysis Agent", symptom_agent),
        ("Medical Knowledge Agent", medical_agent),
        ("Knowledge Answer Agent", knowledge_agent),
    ],
    COMPARE_CONDITIONS: [
        ("Patient Profile Agent", profile_agent),
        ("Symptom Analysis Agent", symptom_agent),
        ("Comparison Agent", compare_agent),
    ],
    MEDICATION_SAFETY: [
        ("Patient Profile Agent", profile_agent),
        ("Medication Safety Agent", medication_safety_agent),
    ],
    OUT_OF_SCOPE: [
        ("Out-of-Scope Agent", out_of_scope_agent),
    ],
}

# What the state must say before each non-triage chain runs, so downstream
# agents and the UI never treat an information question as a symptom report.
OPERATION_PRESETS = {
    RETRIEVE_KNOWLEDGE: {
        "query_type": "knowledge_question",
        "risk_level": "INFO",
        "risk": "Risk level: INFO. General question, not a symptom report.",
        "recommendation": (
            "This is general information. Discuss anything that applies to you "
            "with a doctor."
        ),
        "hospital_navigation": "Not applicable for a general question.",
    },
    COMPARE_CONDITIONS: {
        "query_type": "comparison_question",
        "risk_level": "INFO",
        "risk": "Risk level: INFO. Comparison question, not a symptom report.",
        "recommendation": (
            "This is general information. Discuss anything that applies to you "
            "with a doctor."
        ),
        "hospital_navigation": "Not applicable for a general question.",
    },
    MEDICATION_SAFETY: {
        "query_type": "medication_request",
        "risk_level": "INFO",
        "risk": "Risk level: INFO. Prescription request, declined by design.",
    },
    OUT_OF_SCOPE: {
        "query_type": "out_of_scope",
        "risk_level": "INFO",
        "risk": "Risk level: INFO. Not a medical question.",
    },
    TRIAGE_SYMPTOMS: {"query_type": "symptom_report"},
}


def emergency_state(patient, state, start_time):
    state["profile"] = dict(patient)
    state["symptoms"] = patient["symptoms"]
    state["symptom_details"] = {"identified_symptoms": [], "duration_days": None}
    state["medical_context"] = "Skipped: emergency fast-path."
    state["query_type"] = "emergency"
    state["risk_level"] = "EMERGENCY"
    state["risk"] = "Risk level: EMERGENCY. Safety screen triggered."
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
    state["agent_log"].append(
        "Emergency Fast-Path executed - remaining agents skipped for speed."
    )
    state["response_time_seconds"] = round(time.perf_counter() - start_time, 2)
    return state


def run_medguide(patient, on_step=None):
    """Runs the multi-agent pipeline.

    The planner agent chooses which operation to run first; only that chain
    executes. on_step: optional callback(step_name, index, total) invoked right
    before each agent runs, so the UI can show live progress.
    """
    start_time = time.perf_counter()
    state = {"patient": patient, "agent_log": []}

    query = patient.get("symptoms", "")
    decision = plan(query)
    operation = decision["operation"]

    state["operation"] = operation
    state["planner_decision"] = decision
    state["agent_log"].append(
        "Planner Agent chose {} (model said {} at confidence {:.2f}{}).".format(
            operation,
            decision["predicted_operation"],
            decision["confidence"],
            "; overridden by " + ", ".join(decision["overrides"])
            if decision["overrides"] else "",
        )
    )

    if operation == EMERGENCY_ESCALATE:
        return emergency_state(patient, state, start_time)

    state.update(OPERATION_PRESETS.get(operation, {}))
    steps = OPERATION_CHAINS[operation]

    total = len(steps)
    for i, (step_name, agent) in enumerate(steps, start=1):
        if on_step:
            on_step(step_name, i, total)
        state = agent(state)

    state["response_time_seconds"] = round(time.perf_counter() - start_time, 2)
    return state
