import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import planner  # noqa: E402
from agents.compare_agent import asked_aspect, extract_sides  # noqa: E402
from workflow import OPERATION_CHAINS, run_medguide  # noqa: E402

PATIENT = {
    "age": 22,
    "gender": "Female",
    "medical_history": "None reported",
    "allergies": "None reported",
    "medications": "None reported",
    "location": "Chennai",
}


def ask(text):
    return run_medguide({**PATIENT, "symptoms": text})


def chosen(text):
    return planner.plan(text)["operation"]


def test_every_operation_has_a_chain_or_is_the_emergency_fast_path():
    covered = set(OPERATION_CHAINS) | {planner.EMERGENCY_ESCALATE}
    assert covered == set(planner.OPERATIONS)


def test_planner_picks_the_operation_each_query_needs():
    assert chosen("severe chest pain and cannot breathe") == planner.EMERGENCY_ESCALATE
    assert chosen("i have fever and cough for three days") == planner.TRIAGE_SYMPTOMS
    assert chosen("What causes Kidney Disease ?") == planner.RETRIEVE_KNOWLEDGE
    assert chosen("what is the difference between asthma and bronchitis") == (
        planner.COMPARE_CONDITIONS
    )
    assert chosen("which tablet should i take for fever") == planner.MEDICATION_SAFETY
    assert chosen("who won the world cup last year") == planner.OUT_OF_SCOPE


def test_safety_overrides_win_over_the_model():
    # Emergency wording phrased as an information question must still escalate.
    decision = planner.plan("what is the treatment for crushing chest pain and sweating")
    assert decision["operation"] == planner.EMERGENCY_ESCALATE
    assert decision["overrides"]

    decision = planner.plan("what is the dosage of paracetamol for a child")
    assert decision["operation"] == planner.MEDICATION_SAFETY
    assert decision["overrides"]


def test_overrides_never_make_the_plan_less_cautious():
    assert planner.safest(planner.OUT_OF_SCOPE, planner.EMERGENCY_ESCALATE) == (
        planner.EMERGENCY_ESCALATE
    )
    assert planner.safest(planner.EMERGENCY_ESCALATE, planner.MEDICATION_SAFETY) == (
        planner.EMERGENCY_ESCALATE
    )


def test_medication_request_is_declined_without_naming_a_drug():
    state = ask("which antibiotic should i take for throat infection")

    assert state["operation"] == planner.MEDICATION_SAFETY
    assert state["ai_used"] is False
    answer = state["final_response"].lower()
    assert "cannot tell you which medicine" in answer
    for drug in ("amoxicillin", "azithromycin", "paracetamol", " mg"):
        assert drug not in answer


def test_out_of_scope_query_is_declined_and_no_chain_runs():
    state = ask("write me a python program to sort a list")

    assert state["operation"] == planner.OUT_OF_SCOPE
    assert "outside what I can help with" in state["final_response"]
    assert not state.get("medquad_hits")


def test_comparison_question_retrieves_both_sides():
    state = ask("what is the difference between asthma and bronchitis")

    assert state["operation"] == planner.COMPARE_CONDITIONS
    assert len(state["medquad_hits"]) == 2
    topics = {topic.lower() for topic in state["compare_sides"]}
    assert any("asthma" in topic for topic in topics)
    assert any("bronchitis" in topic for topic in topics)


def test_comparison_sides_survive_an_aspect_in_the_question():
    assert extract_sides("compare the symptoms of dengue and malaria") == (
        "dengue",
        "malaria",
    )
    assert asked_aspect("compare the symptoms of dengue and malaria") == "symptoms"
    assert extract_sides("what causes diabetes") is None


def test_planner_decision_is_recorded_for_every_query():
    state = ask("What causes Kidney Disease ?")
    decision = state["planner_decision"]

    assert decision["operation"] == planner.RETRIEVE_KNOWLEDGE
    assert 0.0 <= decision["confidence"] <= 1.0
    assert any("Planner Agent chose" in line for line in state["agent_log"])
