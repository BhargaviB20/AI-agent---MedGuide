from ml.predictors import predict_intent
from rag.meddialog_retriever import retrieve_meddialog
from rag.retriever import retrieve_medquad

# Words appended to the retrieval query once the intent classifier has decided
# what the patient is actually asking for, so the query matches the phrasing
# MedQuAD uses for that kind of question.
INTENT_CUES = {
    "symptoms": ("symptom", "sign"),
    "treatment": ("treat", "therap", "manage"),
    "causes": ("cause",),
    "prevention": ("prevent",),
    "diagnosis": ("diagnos", "test"),
    "genetics": ("inherit", "genetic"),
    "frequency": ("how many people",),
    "prognosis": ("outlook", "prognos"),
    "definition": ("what is", "what are"),
}

# TF-IDF alone ranks any passage about the right disease highly, so a "causes"
# question often lands on the "what is" passage. The bonus only applies to
# passages on the same topic as the best hit, so it re-orders within a topic
# instead of pulling in a different disease that matches the wording.
CUE_BONUS = 0.5

# Some MedQuAD rows are only lists of external links, which tell a patient
# nothing, so they are dropped whenever explanatory text is also available.
BOILERPLATE_MARKERS = (
    "these resources address",
    "genetic testing registry",
    "gene review",
)


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


def asked_intent(text, predicted, confidence):
    """The wording of the question is more trustworthy than the classifier, so
    it wins when both suggest an intent."""
    lowered = (text or "").lower()
    for intent, cues in INTENT_CUES.items():
        if any(cue in lowered for cue in cues):
            return intent
    return predicted if confidence >= 0.5 else None


def is_boilerplate(hit):
    answer = (hit["answer"] or "").lower()[:400]
    return any(marker in answer for marker in BOILERPLATE_MARKERS)


def rerank_by_intent(hits, intent):
    if not hits:
        return hits

    useful = [hit for hit in hits if not is_boilerplate(hit)]
    ranked = useful or hits

    cues = INTENT_CUES.get(intent, ()) if intent else ()
    primary_topic = (ranked[0]["focus_area"] or "").lower()

    def sort_key(hit):
        question = (hit["question"] or "").lower()
        on_topic = (hit["focus_area"] or "").lower() == primary_topic
        bonus = (
            CUE_BONUS
            if on_topic and any(cue in question for cue in cues)
            else 0.0
        )
        return hit["score"] + bonus

    return sorted(ranked, key=sort_key, reverse=True)


def medical_agent(state):
    """Retrieves supporting context from BOTH datasets:
    MedQuAD (reference medical Q&A) and MedDialog (real doctor replies)."""
    symptoms = state["patient"]["symptoms"]

    intent, intent_confidence = predict_intent(symptoms)
    state["query_intent"] = intent
    state["query_intent_confidence"] = round(intent_confidence, 4)

    effective_intent = asked_intent(symptoms, intent, intent_confidence)

    query = symptoms
    if effective_intent:
        query = f"{symptoms} {INTENT_QUERY_TERMS.get(effective_intent, '')}".strip()

    blocks = []

    try:
        hits = rerank_by_intent(
            retrieve_medquad(query, top_k=30), effective_intent
        )[:3]
        state["medquad_hits"] = hits
        for item in hits:
            blocks.append(
                f"Reference topic: {item['focus_area'] or item['question']}\n"
                f"Reference information: {item['answer']}"
            )
    except Exception as e:
        state["medquad_hits"] = []
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
