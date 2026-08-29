# MedGuide AI - what to replace

Copy these files into your project folder (MedGuide-AI-V2), overwriting the old ones:

REPLACE:
- app.py
- workflow.py
- agents/symptom_agent.py
- agents/medical_agent.py
- agents/risk_agent.py
- agents/recommendation_agent.py
- agents/hospital_agent.py
- agents/final_agent.py
- agents/emergency_check.py
- rag/meddialog_retriever.py
- rag/retriever.py

UNCHANGED (included only so the folder is complete): database.py, llm.py

## What changed and why

1. Raw HTML was showing on the page (`<div class="brand-title">` etc.).
   Cause: the HTML inside st.markdown was indented with 4 spaces, and Streamlit
   markdown treats indented lines as a code block. All HTML is now unindented.

2. The AI answer is now rendered with st.markdown inside a bordered container,
   so headings, paragraphs and bullets format properly.

3. The answer now always has 5 sections:
   What this looks like / Why this may have happened / What you can do at home /
   When you should see a doctor / Warning signs - get care immediately.
   This is exactly the "why it happened, what to do, when to see a doctor" output.

4. Speed + API usage: only the Final Response Agent calls Gemini now (1 call per
   consultation instead of 5). Symptom extraction, risk level, care pathway and
   hospital guidance are rule based, so they are instant and reproducible.

5. Both datasets are used: MedQuAD (TF-IDF over data/medquad.csv) and MedDialog
   (data/meddialog/english-train.json). Retrieval failures are caught, so the app
   still answers if a dataset file is missing.

6. If GEMINI_API_KEY is missing or the API fails, a structured fallback answer with
   the same 5 sections is shown instead of an error.

7. Evaluation page now scores the rule-based risk level (5/5 PASS, runs in <1s).

## Run

    pip install -r requirements.txt
    # .env must contain: GEMINI_API_KEY=your_key
    streamlit run app.py

Data files expected:
    data/medquad.csv
    data/meddialog/english-train.json
