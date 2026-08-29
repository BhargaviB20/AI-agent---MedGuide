from rag.meddialog_retriever import retrieve_meddialog
from rag.retriever import retrieve_medquad


def medical_agent(state):
    """Retrieves supporting context from BOTH datasets:
    MedQuAD (reference medical Q&A) and MedDialog (real doctor replies)."""
    symptoms = state["patient"]["symptoms"]

    blocks = []

    try:
        for item in retrieve_medquad(symptoms, top_k=3):
            blocks.append(
                f"Reference topic: {item['focus_area'] or item['question']}\n"
                f"Reference information: {item['answer']}"
            )
    except Exception as e:
        print(f"[medical_agent] MedQuAD retrieval failed: {e}")

    try:
        for item in retrieve_meddialog(symptoms, top_k=3):
            blocks.append(
                f"Similar patient case: {item['patient']}\n"
                f"Doctor response: {item['doctor']}"
            )
    except Exception as e:
        print(f"[medical_agent] MedDialog retrieval failed: {e}")

    state["medical_context"] = (
        "\n\n".join(blocks) if blocks else "No closely matching medical information found."
    )

    state["agent_log"].append("Medical Knowledge Retrieval Agent completed.")
    return state
