import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.risk_agent import risk_agent, rule_based_risk_level
from agents.symptom_agent import symptom_agent
from ml.predictors import most_urgent


def test_most_urgent_prefers_the_more_urgent_level():
    assert most_urgent("LOW", "EMERGENCY") == "EMERGENCY"
    assert most_urgent("MODERATE", "HIGH") == "HIGH"
    assert most_urgent(None, "LOW") == "LOW"
    assert most_urgent(None, None) is None


def test_rule_ladder_flags_red_flag_keywords():
    level, _ = rule_based_risk_level(
        {"age": 30, "symptoms": "severe chest pain and sweating"}, {}
    )
    assert level == "EMERGENCY"


def test_red_flag_case_is_never_de_escalated_by_the_classifier():
    state = {"patient": {"age": 30, "symptoms": "coughing blood since morning"},
             "agent_log": []}
    state = risk_agent(symptom_agent(state))
    assert state["risk_level"] == "EMERGENCY"


def test_risk_agent_reports_which_component_decided():
    state = {"patient": {"age": 25, "symptoms": "mild sneezing since this morning"},
             "agent_log": []}
    state = risk_agent(symptom_agent(state))
    detail = state["risk_detail"]
    assert detail["final_level"] in {"LOW", "MODERATE", "HIGH", "EMERGENCY"}
    assert detail["rule_based_level"] in {"LOW", "MODERATE", "HIGH", "EMERGENCY"}
