from ml.predictors import predict_intent
from rag.meddialog_retriever import retrieve_meddialog
from rag.retriever import retrieve_medquad

# Words appended to the retrieval query once the intent classifier has decided
# what the patient is actually asking for, so the query matches the phrasing
# MedQuAD uses for that kind of question.
INTENT_QUERY_TERMS = {
    "symptoms": "symptoms signs",
    "treatment": "treatments treatment management",
    "causes": "causes",
    "prevention": "prevention prevent",
    "diagnosis": "diagnosis diagnosed tests",
    "genetics": "inherited genetic",
    "frequency": "how many people affected",
    "prognosis": "outlook prognosis",
    "definition": "what is",
}


def medical_agent(state):
    """Retrieves supporting context from BOTH datasets:
    MedQuAD (reference medical Q&A) and MedDialog (real doctor replies)."""
    symptoms = state["patient"]["symptoms"]

    intent, intent_confidence = predict_intent(symptoms)
    state["query_intent"] = intent
    state["query_intent_confidence"] = round(intent_confidence, 4)

    query = symptoms
    if intent and intent_confidence >= 0.5:
        query = f"{symptoms} {INTENT_QUERY_TERMS.get(intent, '')}".strip()

    blocks = []

    try:
        for item in retrieve_medquad(query, top_k=3):
            blocks.append(
                f"Reference topic: {item['focus_area'] or item['question']}\n"
                f"Reference information: {item['answer']}"
            )
    except Exception as e:
        print(f"[medical_agent] MedQuAD retrieval failed: {e}")

    try:
        for item in retrieve_meddialog(query, top_k=3):
            blocks.append(
                f"Similar patient case: {item['patient']}\n"
                f"Doctor response: {item['doctor']}"
            )
    except Exception as e:
        print(f"[medical_agent] MedDialog retrieval failed: {e}")

    state["medical_context"] = (
        "\n\n".join(blocks) if blocks else "No closely matching medical information found."
    )

    log = "Medical Knowledge Retrieval Agent completed."
    if intent:
        log += f" Query intent: {intent} ({intent_confidence:.2f})."
    state["agent_log"].append(log)
    return state
