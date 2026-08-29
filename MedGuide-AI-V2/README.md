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

## Trained components and measured results

Two supervised models are trained from the datasets in `data/` and evaluated on
held-out data: a triage classifier (used by the Risk Agent, with the red-flag
keyword screen kept as a guardrail) and a MedQuAD question-intent classifier
(used by the Retrieval Agent). Retrieval and the full pipeline are evaluated
too.

See [TRAINING_AND_EVALUATION.md](TRAINING_AND_EVALUATION.md) for the datasets,
protocols, commands and the measured numbers. Raw metrics live in `results/`
and are displayed on the app's Evaluation page.

## Safety

This is an academic prototype. It does not diagnose, prescribe, or provide real-time
hospital availability. Use verified medical sources and qualified clinicians for real care.
