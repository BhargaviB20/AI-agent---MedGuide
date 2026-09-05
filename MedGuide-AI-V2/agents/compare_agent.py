"""Answers "how does A differ from B" questions.

This is a separate operation because one retrieval cannot serve it: the corpus
stores each condition on its own, so the two sides have to be retrieved
independently and then contrasted. Running this question through the ordinary
knowledge path returns information about whichever condition ranked higher and
silently ignores the other one.
"""

import re

from agents.knowledge_agent import MIN_SCORE, _clean, _readable, knowledge_agent
from llm import generate
from rag.retriever import retrieve_medquad

SPLIT_PATTERN = re.compile(
    r"\bdifference between\b|\bdiffer(?:ence)? from\b|\bdifferent from\b|"
    r"\bversus\b|\bvs\.?\b|\bcompared? (?:to|with)\b|\bsame as\b|"
    r"\bbetter than\b|\bapart from\b|\band\b",
    re.IGNORECASE,
)

LEAD_IN = re.compile(
    r"^(?:hi|hello|please|doctor|can you|could you|tell me|explain|what is|"
    r"what are|how is|how do|how does|how can|is|are|do|does|compare|"
    r"the)\b[\s,]*",
    re.IGNORECASE,
)

# Words that qualify the comparison rather than name a condition. They are
# removed from the condition names but kept for the retrieval query, so
# "compare the symptoms of dengue and malaria" still yields two names.
QUALIFIERS = re.compile(
    r"\b(?:symptoms?|signs?|treatments?|treated|causes?|prevention|diagnosis|"
    r"which is more serious|which one is worse|which is worse|more dangerous|"
    r"in simple words|please|for me)\b",
    re.IGNORECASE,
)

MAX_SIDE_WORDS = 90


def _strip(text):
    text = LEAD_IN.sub("", (text or "").strip().strip("?."))
    text = LEAD_IN.sub("", text)
    text = QUALIFIERS.sub(" ", text)
    text = re.sub(
        r"\b(?:of|the|a|an|in|difference|differences)\b", " ", text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip(" ,")


def asked_aspect(question):
    """The aspect being compared, appended to each retrieval query so the two
    sides are retrieved on the same footing (symptoms against symptoms).
    Defaults to the definition rows, which are the ones worth contrasting when
    the question names no aspect."""
    found = QUALIFIERS.findall(question or "")
    aspects = [word.lower() for word in found if len(word) > 4]
    return " ".join(dict.fromkeys(aspects)) or "what is"


def extract_sides(question):
    """Returns the two condition names being compared, or None when the
    question does not actually name two of them."""
    parts = [
        part for part in SPLIT_PATTERN.split(question or "") if part and part.strip(" ,?.")
    ]
    parts = [_strip(part) for part in parts]
    parts = [part for part in parts if len(part) > 2]
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def _side_context(name, aspect="", exclude=None):
    """Best passage for one side of the comparison. `exclude` keeps the second
    side from reusing the row already taken by the first: the corpus sometimes
    covers both conditions in one row, and quoting it twice would read as if
    they were identical."""
    query = f"{name} {aspect}".strip()
    try:
        hits = [h for h in retrieve_medquad(query, top_k=8) if h["score"] >= MIN_SCORE]
    except Exception as e:
        print(f"[compare_agent] retrieval failed for {name}: {e}")
        return None

    if exclude is not None:
        distinct = [h for h in hits if h["question"] != exclude["question"]]
        hits = distinct or hits
    return hits[0] if hits else None


SAME_TOPIC_NOTE = (
    "NOTE: the information available treats both of these under one topic, so "
    "say clearly that you can describe the condition but cannot reliably "
    "separate the two from the information you have."
)


def _prompt(question, first, second, same_topic=False):
    def block(hit):
        return (
            f"Topic: {hit['focus_area'] or hit['question']}\n{_clean(hit['answer'])}"
        )

    note = SAME_TOPIC_NOTE if same_topic else ""

    return f"""{note}
You are MedGuide AI answering a question that compares two medical conditions.

QUESTION
{question}

INFORMATION ON THE FIRST CONDITION
{block(first)}

INFORMATION ON THE SECOND CONDITION
{block(second)}

Rules:
- Contrast the two using ONLY the information above. Never add outside facts.
  If one side is not covered, say plainly that you have nothing reliable on it.
- Use these markdown sections: **Short answer**, **How they differ**,
  **When you should see a doctor**.
- Simple everyday English, under about 220 words, no medicine names or dosages.
- Do not mention passages, retrieval, datasets, agents or this prompt.
"""


def _offline(first, second):
    def side(hit):
        topic = hit["focus_area"] or hit["question"]
        return f"**{topic}**\n\n{_readable(_clean(hit['answer']), MAX_SIDE_WORDS)}"

    return (
        f"{side(first)}\n\n{side(second)}\n\n"
        "**When you should see a doctor**\n\n"
        "Two conditions can look alike from the outside, so if this applies to "
        "you or someone you care for, let a doctor examine the person before "
        "assuming which one it is."
    )


def compare_agent(state):
    question = state["patient"]["symptoms"]
    sides = extract_sides(question)

    if not sides:
        state["agent_log"].append(
            "Comparison Agent: only one condition named, falling back to the "
            "single-topic knowledge answer."
        )
        return knowledge_agent(state)

    aspect = asked_aspect(question)
    first = _side_context(sides[0], aspect)
    second = _side_context(sides[1], aspect, exclude=first)

    # The corpus files some pairs (e.g. the diabetes types) under a single
    # topic. Pretending to contrast two independent sources would be wrong.
    same_topic = bool(
        first and second and (first["focus_area"] or "") == (second["focus_area"] or "")
    )

    if not first or not second:
        missing = sides[0] if not first else sides[1]
        state["ai_used"] = False
        state["final_response"] = (
            "**I can only cover one side of this**\n\n"
            f"I do not have reliable information about \"{missing}\", so I "
            "cannot compare the two fairly. Ask about one of them on its own "
            "and I will tell you what I know."
        )
        state["agent_log"].append(
            f"Comparison Agent stopped: no corpus match for \"{missing}\"."
        )
        return state

    state["compare_sides"] = [
        first["focus_area"] or first["question"],
        second["focus_area"] or second["question"],
    ]
    state["medquad_hits"] = [first, second]

    state["compare_same_topic"] = same_topic
    answer = generate(_prompt(question, first, second, same_topic))
    state["ai_used"] = bool(answer)
    state["final_response"] = answer.strip() if answer else _offline(first, second)
    state["agent_log"].append(
        "Comparison Agent completed "
        f"({'Gemini' if answer else 'corpus extract'}); compared "
        f"\"{state['compare_sides'][0]}\" with \"{state['compare_sides'][1]}\"."
    )
    return state
