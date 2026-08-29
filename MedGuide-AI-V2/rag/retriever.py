import csv
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "medquad.csv"


@lru_cache(maxsize=1)
def load_medquad():
    if not DATA_FILE.exists():
        print(f"[medquad] Dataset not found at {DATA_FILE}")
        return [], None, None

    docs = []

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            question = row.get("question", "") or ""
            answer = row.get("answer", "") or ""
            source = row.get("source", "") or ""
            focus_area = row.get("focus_area", "") or ""

            if not answer.strip():
                continue

            docs.append({
                "text": f"{question} {answer} {focus_area}",
                "question": question,
                "answer": answer,
                "source": source,
                "focus_area": focus_area,
            })

    if not docs:
        return [], None, None

    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    matrix = vectorizer.fit_transform([doc["text"] for doc in docs])

    return docs, vectorizer, matrix


def retrieve_medquad(query, top_k=3):
    """Returns a list of dicts from MedQuAD ranked by TF-IDF similarity."""
    docs, vectorizer, matrix = load_medquad()
    if not docs:
        return []

    scores = cosine_similarity(vectorizer.transform([query]), matrix).flatten()
    ranked = scores.argsort()[::-1][:top_k]

    return [
        {
            "question": docs[i]["question"],
            "answer": docs[i]["answer"][:1200],
            "focus_area": docs[i]["focus_area"],
            "source": docs[i]["source"],
            "score": float(scores[i]),
        }
        for i in ranked
        if scores[i] > 0
    ]


def retrieve_context(query, top_k=3):
    """Backward-compatible string version."""
    return "\n\n".join(
        f"**Focus Area:** {r['focus_area']}\n"
        f"**Question:** {r['question']}\n"
        f"**Answer:** {r['answer']}"
        for r in retrieve_medquad(query, top_k)
    )
