"""Handles the two operations whose correct behaviour is to *not* answer.

MEDICATION_SAFETY  someone asks which drug or dose to take. Naming a drug or a
                   dose without an examination is unsafe, so the operation
                   declines and redirects, deterministically - no LLM involved,
                   because a generated answer could slip a drug name in.
OUT_OF_SCOPE       the query is not medical. Answering it would make the
                   system look like a general chatbot and invite medical
                   trust it has not earned.
"""


def medication_safety_agent(state):
    state["ai_used"] = False
    state["risk_level"] = state.get("risk_level") or "INFO"
    state["final_response"] = (
        "**I cannot tell you which medicine or dose to take**\n\n"
        "Choosing a medicine depends on your weight, other illnesses, other "
        "tablets you already take, allergies and pregnancy, and getting it "
        "wrong can be dangerous. That decision needs a doctor or a pharmacist "
        "who can check those things.\n\n"
        "**What you can do instead**\n\n"
        "Describe what you are feeling and for how long, and I will tell you "
        "what it could be about, what helps without medicines, and how soon "
        "you should be seen.\n\n"
        "**Get care urgently if**\n\n"
        "You have chest pain, trouble breathing, heavy bleeding, fainting, "
        "confusion, or a fever with a stiff neck."
    )
    state["recommendation"] = (
        "Ask a doctor or pharmacist before taking any medicine, and carry the "
        "list of anything you already take."
    )
    state["hospital_navigation"] = (
        "Any nearby clinic or pharmacy can advise on medicines; go to an "
        "emergency department only if you have the warning signs listed above."
    )
    state["agent_log"].append(
        "Medication Safety Agent completed (prescription request declined "
        "deterministically, no model call)."
    )
    return state


def out_of_scope_agent(state):
    state["ai_used"] = False
    state["risk_level"] = state.get("risk_level") or "INFO"
    state["final_response"] = (
        "**This is outside what I can help with**\n\n"
        "I only answer health questions: what a condition is, why symptoms "
        "happen, what helps, and when to see a doctor.\n\n"
        "**Try something like**\n\n"
        "\"I have had a cough and mild fever for four days, what should I "
        "do?\" or \"what causes low blood pressure?\""
    )
    state["recommendation"] = "Not applicable - the question is not medical."
    state["hospital_navigation"] = "Not applicable."
    state["agent_log"].append("Out-of-Scope Agent completed (query declined).")
    return state
