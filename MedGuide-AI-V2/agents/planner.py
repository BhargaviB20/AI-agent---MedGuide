"""Operation-selection agent (the Master Orchestrator's decision step).

MedGuide can run several different operations over the corpora, and they are
not interchangeable: triaging a general information question produces canned
advice, answering a red flag as information is unsafe, and answering "which
tablet should I take" at all is unsafe. So the orchestrator has to *choose*
the operation before any of the downstream agents run.

Three policies are implemented so the choice can be measured against
baselines (see ml/eval_planner.py):

    rule_plan      hand-written keyword policy (the original router)
    model_plan     trained TF-IDF classifier over labelled queries
    llm_plan       Gemini asked to pick an operation, no medical content

The deployed policy is `plan`, which is model_plan wrapped in deterministic
safety overrides: red-flag wording always forces EMERGENCY_ESCALATE and a
prescription request always forces MEDICATION_SAFETY, whatever the model said.
Overrides can only move the plan towards the safer operation, never away.
"""

import re

from agents.emergency_check import detect_emergency
from agents.query_router import is_knowledge_question
from ml.predictors import predict_operation

EMERGENCY_ESCALATE = "EMERGENCY_ESCALATE"
TRIAGE_SYMPTOMS = "TRIAGE_SYMPTOMS"
RETRIEVE_KNOWLEDGE = "RETRIEVE_KNOWLEDGE"
COMPARE_CONDITIONS = "COMPARE_CONDITIONS"
MEDICATION_SAFETY = "MEDICATION_SAFETY"
OUT_OF_SCOPE = "OUT_OF_SCOPE"

OPERATIONS = (
    EMERGENCY_ESCALATE,
    TRIAGE_SYMPTOMS,
    RETRIEVE_KNOWLEDGE,
    COMPARE_CONDITIONS,
    MEDICATION_SAFETY,
    OUT_OF_SCOPE,
)

# Safest first. Used so an override never makes the plan less cautious.
SAFETY_ORDER = (
    EMERGENCY_ESCALATE,
    MEDICATION_SAFETY,
    TRIAGE_SYMPTOMS,
    COMPARE_CONDITIONS,
    RETRIEVE_KNOWLEDGE,
    OUT_OF_SCOPE,
)

COMPARISON_MARKERS = (
    r"difference between",
    r"differ from",
    r"different from",
    r"\bversus\b",
    r"\bvs\b",
    r"\bvs\.",
    r"compare[ds]? (?:the )?",
    r"comparison between",
    r"same as",
    r"better than",
)

# Asking for a named drug, a dose or a prescription. Answering these is out of
# scope by design, so the request has to be recognised rather than retrieved.
MEDICATION_MARKERS = (
    r"which (?:tablet|medicine|medication|drug|antibiotic|pill|syrup|injection)",
    r"what (?:tablet|medicine|medication|drug|antibiotic|pill|syrup|dose|dosage)",
    r"\b(?:how many|how much) (?:mg|ml|tablets|pills)",
    r"\bdosage\b",
    r"\bdose of\b",
    r"\bprescribe\b",
    r"\bprescription\b",
    r"can i take \w+",
    r"should i take \w+",
    r"\bmg of\b",
    r"name of the (?:medicine|tablet|drug)",
    r"suggest (?:a |some )?(?:medicine|tablet|drug|antibiotic)",
)

# Everyday non-medical requests. Kept deliberately narrow: anything that could
# be a health question must not land here.
OUT_OF_SCOPE_MARKERS = (
    r"\bcapital of\b",
    r"\bweather\b",
    r"\bfootball\b|\bcricket\b|\bworld cup\b|\bmatch score\b",
    r"write (?:me )?(?:a |an )?(?:poem|song|essay|code|program)",
    r"\bpython\b|\bjavascript\b|\bexcel\b|\blaptop\b|\bwifi\b|\bphone\b",
    r"\bstock price\b|\bbitcoin\b",
    r"\brecipe\b",
    r"\btranslate\b",
    r"who (?:is|was) the (?:president|prime minister)",
    r"\bmovie\b|\bsong\b",
    r"\bflight\b|\btrain ticket\b|\bhotel\b",
    r"\bhomework\b|\bexam timetable\b",
)

COMPARISON_PATTERN = re.compile("|".join(COMPARISON_MARKERS))
MEDICATION_PATTERN = re.compile("|".join(MEDICATION_MARKERS))
OUT_OF_SCOPE_PATTERN = re.compile("|".join(OUT_OF_SCOPE_MARKERS))

MIN_MODEL_CONFIDENCE = 0.35


def safest(*operations):
    known = [op for op in operations if op in SAFETY_ORDER]
    if not known:
        return None
    return min(known, key=SAFETY_ORDER.index)


def rule_plan(text):
    """Keyword policy. This is the baseline the trained planner is compared to,
    and it is also the fallback when no model has been trained yet."""
    cleaned = (text or "").strip().lower()

    if not cleaned:
        return OUT_OF_SCOPE
    if detect_emergency(cleaned):
        return EMERGENCY_ESCALATE
    if MEDICATION_PATTERN.search(cleaned):
        return MEDICATION_SAFETY
    if OUT_OF_SCOPE_PATTERN.search(cleaned):
        return OUT_OF_SCOPE
    if COMPARISON_PATTERN.search(cleaned):
        return COMPARE_CONDITIONS
    if is_knowledge_question(cleaned):
        return RETRIEVE_KNOWLEDGE
    return TRIAGE_SYMPTOMS


def model_plan(text):
    """Trained planner only. Returns (operation, confidence); falls back to the
    rule policy when the model is missing or unsure."""
    operation, confidence = predict_operation(text)
    if operation in OPERATIONS and confidence >= MIN_MODEL_CONFIDENCE:
        return operation, confidence
    return rule_plan(text), confidence


LLM_PLANNER_PROMPT = """You are the routing component of a medical assistant.
Choose exactly ONE operation for the user message. Answer with the operation
name only, nothing else.

EMERGENCY_ESCALATE  the message describes a possible medical emergency
TRIAGE_SYMPTOMS     the user describes their own symptoms and wants advice
RETRIEVE_KNOWLEDGE  a general information question about a condition
COMPARE_CONDITIONS  asks how two conditions or treatments differ
MEDICATION_SAFETY   asks which medicine, drug or dose to take
OUT_OF_SCOPE        not a medical question at all

USER MESSAGE
{message}
"""


def llm_plan(text):
    """Gemini as the planner, for comparison against the trained planner.
    Returns None when the API is unavailable, so callers can skip it."""
    from llm import generate

    reply = generate(LLM_PLANNER_PROMPT.format(message=text or ""))
    if not reply:
        return None

    upper = reply.strip().upper()
    for operation in OPERATIONS:
        if operation in upper:
            return operation
    return None


def plan(text):
    """Deployed policy: trained planner plus deterministic safety overrides.

    Returns a dict describing the decision so the UI and the evaluation
    scripts can see *why* an operation was chosen, not only which one.
    """
    cleaned = (text or "").strip().lower()
    predicted, confidence = model_plan(text)

    overrides = []
    operation = predicted

    if detect_emergency(cleaned):
        overrides.append("red-flag wording")
        operation = safest(operation, EMERGENCY_ESCALATE)
    if MEDICATION_PATTERN.search(cleaned):
        overrides.append("prescription request")
        operation = safest(operation, MEDICATION_SAFETY)

    return {
        "operation": operation,
        "predicted_operation": predicted,
        "confidence": round(float(confidence), 4),
        "overrides": overrides,
    }
