"""Decides whether the user is describing their own symptoms or asking a
general medical question. The two need different handling: symptom reports go
through triage, information questions must be answered from the corpus."""

import re

from agents.emergency_check import detect_emergency

QUESTION_WORDS = (
    "what", "why", "how", "who", "which", "when", "where",
    "is", "are", "does", "do", "can", "should", "tell me",
)

# First-person wording that means the user is reporting their own condition.
# Matched on word boundaries: "syndrome" must not count as "me".
PERSONAL_MARKERS = (
    r"i have", r"i am", r"i'm", r"\bim\b", r"i've", r"\bive\b", r"i feel",
    r"i felt", r"i got", r"i had", r"\bmy\b", r"\bmine\b",
    r"\bmyself\b", r"for the past", r"since yesterday", r"since morning",
    r"\bage\b", r"years old", r"yrs old", r"y/o",
)

PERSONAL_PATTERN = re.compile("|".join(PERSONAL_MARKERS))


def is_knowledge_question(text):
    """True when the input reads like a general medical information question
    (the kind MedQuAD answers) rather than a personal symptom report.

    Emergency wording always wins, so a red-flag phrased as a question still
    goes down the safety path.
    """
    cleaned = (text or "").strip().lower()

    if not cleaned or detect_emergency(cleaned):
        return False

    if PERSONAL_PATTERN.search(cleaned):
        return False

    first_word = cleaned.split()[0].strip("(),.")
    starts_as_question = (
        first_word in QUESTION_WORDS
        or cleaned.startswith("tell me")
        or cleaned.startswith("what is (are)")
    )

    return starts_as_question or cleaned.endswith("?")
