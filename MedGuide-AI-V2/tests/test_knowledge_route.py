import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.medical_agent import rerank_by_intent  # noqa: E402
from agents.query_router import is_knowledge_question  # noqa: E402
from workflow import run_medguide  # noqa: E402

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


def test_dataset_style_questions_are_routed_as_knowledge_questions():
    for question in (
        "What causes Urinary Incontinence ?",
        "What is (are) Glaucoma ?",
        "How is asthma treated ?",
        "Is Alagille syndrome inherited ?",
        "How many people are affected by cystic fibrosis ?",
        "tell me about diabetes",
    ):
        assert is_knowledge_question(question), question


def test_personal_symptom_reports_are_not_knowledge_questions():
    for report in (
        "age 22 cold and fever and headache for 7 days",
        "i have a sore throat since yesterday",
        "my stomach hurts after eating",
        "im feeling dizzy",
    ):
        assert not is_knowledge_question(report), report


def test_disease_names_containing_me_are_not_read_as_first_person():
    # "syndrome" contains "me"; substring matching used to misroute these.
    assert is_knowledge_question("What are the symptoms of Marfan syndrome ?")
    assert is_knowledge_question("Is Alagille syndrome inherited ?")


def test_red_flag_question_still_goes_down_the_emergency_path():
    assert not is_knowledge_question("what should i do for severe chest pain ?")
    state = ask("severe chest pain and cannot breathe")
    assert state["risk_level"] == "EMERGENCY"


def test_knowledge_answer_quotes_the_matching_medquad_row():
    state = ask("Is Alagille syndrome inherited ?")

    assert state["query_type"] == "knowledge_question"
    assert state["risk_level"] == "INFO"

    hits = state["medquad_hits"]
    assert hits[0]["question"].lower() == "is alagille syndrome inherited ?"

    answer = state["final_response"]
    assert "autosomal dominant" in answer.lower()
    assert "MedQuAD" in answer


def test_symptom_report_still_runs_the_full_triage_pipeline():
    state = ask("age 22 cold and fever and headache for 7 days")

    assert state["query_type"] == "symptom_report"
    assert state["risk_level"] in {"LOW", "MODERATE", "HIGH", "EMERGENCY"}
    assert any("Hospital Navigation Agent" in line for line in state["agent_log"])


def test_off_topic_question_says_so_instead_of_inventing_an_answer():
    state = ask("how do i become a pilot ?")
    assert "No matching information" in state["final_response"]


def test_link_only_passages_are_dropped_when_real_text_exists():
    hits = [
        {
            "question": "What are the treatments for allergic asthma ?",
            "answer": "These resources address the diagnosis or management of ...",
            "focus_area": "allergic asthma",
            "source": "GHR",
            "score": 0.9,
        },
        {
            "question": "What are the treatments for Asthma ?",
            "answer": "Asthma is a long-term disease that has no cure ...",
            "focus_area": "Asthma",
            "source": "NHLBI",
            "score": 0.6,
        },
    ]
    assert rerank_by_intent(hits, "treatment")[0]["focus_area"] == "Asthma"
