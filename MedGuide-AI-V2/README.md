# MedGuide AI V2

Multi-Agent healthcare navigation prototype with a Gemini LLM layer and local RAG.

## Setup

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Install:
```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add your Gemini API key.

Run:
```bash
streamlit run app.py
```

## Architecture

User -> Profile Agent -> Symptom Agent -> Medical Retrieval Agent ->
Risk Agent -> Recommendation Agent -> Hospital Navigation Agent -> Final Agent.

## Safety

This is an academic prototype. It does not diagnose, prescribe, or provide real-time
hospital availability. Use verified medical sources and qualified clinicians for real care.
