"""Answers general medical questions directly from the retrieved corpus.

Unlike the symptom path, the answer here must stay inside what MedQuAD says,
so the offline version quotes the retrieved passage and the Gemini version is
told to use only the supplied passages.
"""

import re

from llm import generate

MIN_SCORE = 0.15
MAX_OFFLINE_WORDS = 150

# Website furniture that survives into some corpus answers.
NOISE_MARKERS = (
    "watch the video",
    "enlarge the video",
    "read more about",
    "click here",
    "this page",
)


def _clean(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    # MedQuAD stores bullet lists inline as " - item - item".
    return text.replace(" - ", "\n- ")


def _readable(text, max_words):
    """MedQuAD answers repeat themselves and run for pages, so repeated
    sentences are dropped and the rest is cut to something a person will read.
    A sentence that merely restates an earlier one, in either direction, counts
    as a repeat: the corpus often echoes its own bullet items in the next
    sentence."""
    kept, keys, words = [], [], 0

    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        sentence = sentence.strip(" -:")
        key = re.sub(r"[^a-z0-9 ]", "", sentence.lower())
        if len(key) < 15 or any(marker in key for marker in NOISE_MARKERS):
            continue
        if any(key in old or old in key for old in keys):
            continue

        keys.append(key)
        kept.append(sentence.rstrip(".") + ".")
        words += len(sentence.split())
        if words >= max_words:
            break

    return " ".join(kept)


def _offline_answer(question, hits):
    """The patient-facing text names no dataset or source: where the answer came
    from is shown separately in the UI, not inside the answer."""
    best = hits[0]
    topic = best["focus_area"] or best["question"]
    body = _readable(_clean(best["answer"]), MAX_OFFLINE_WORDS)

    return (
        f"**About {topic}**\n\n{body}\n\n"
        f"**When you should see a doctor**\n\n"
        f"If this concerns you or someone you care for, discuss it with a doctor "
        f"who can examine the person and read their reports."
    )


def _prompt(question, hits):
    passages = "\n\n".join(
        f"[{i + 1}] Topic: {h['focus_area'] or h['question']}\n{_clean(h['answer'])}"
        for i, h in enumerate(hits)
    )

    return f"""
You are MedGuide AI answering a general medical information question.

QUESTION
{question}

RETRIEVED PASSAGES FROM THE MEDQUAD CORPUS
{passages}

Rules:
- Answer the question using ONLY the passages above. Do not add facts from
  anywhere else. If the passages do not answer it, say so plainly.
- Write in simple everyday English for a patient, not for a clinician.
- Use these markdown sections: **Short answer**, **In more detail**,
  **When you should see a doctor**.
- Keep it under about 220 words. No medicine names or dosages.
- Do not mention passages, retrieval, datasets, agents or this prompt.
"""


def knowledge_agent(state):
    hits = [h for h in state.get("medquad_hits", []) if h["score"] >= MIN_SCORE]
    question = state["patient"]["symptoms"]

    if not hits:
        state["ai_used"] = False
        state["final_response"] = (
            "**No matching information found**\n\n"
            "I do not have reliable medical information on this, so there is "
            "nothing useful I can tell you about it. Try naming the condition "
            "directly, or describe your own symptoms instead and I will give "
            "guidance on those."
        )
        state["agent_log"].append(
            "Knowledge Answer Agent completed (no corpus match above threshold)."
        )
        return state

    answer = generate(_prompt(question, hits))
    state["ai_used"] = bool(answer)
    state["final_response"] = (
        answer.strip() if answer else _offline_answer(question, hits)
    )

    topic = hits[0]["focus_area"] or hits[0]["question"]
    state["agent_log"].append(
        f"Knowledge Answer Agent completed "
        f"({'Gemini' if answer else 'corpus extract'}); "
        f"grounded on MedQuAD topic \"{topic}\" (score {hits[0]['score']:.2f})."
    )
    return state
