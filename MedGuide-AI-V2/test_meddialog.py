//test_meddialog.py

from rag.meddialog_retriever import (
    extract_keywords,
    retrieve_meddialog
)


query = "I have cold, fever and headache for 7 days"

keywords = extract_keywords(query)

print("\nDetected keywords:")
print(keywords)

print("\nSimilar MedDialog cases:\n")

results = retrieve_meddialog(query, top_k=5)

for i, result in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print("Score:", result["score"])
    print("Patient:", result["patient"])
    print("Doctor:", result["doctor"])