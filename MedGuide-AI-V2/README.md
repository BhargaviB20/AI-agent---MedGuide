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

Create `.env` from `.env.example` and add your Gemini API key. Without a key the
app still answers, using the offline fallback.

Run:
```bash
streamlit run app.py
```

## Architecture

A trained **Planner Agent** reads the query and chooses one of six operations;
each operation expands into its own agent chain, so nothing runs a fixed order.
Deterministic safety overrides can only make the choice more cautious.

```
                     +--> EMERGENCY_ESCALATE   immediate escalation, nothing else runs
                     +--> TRIAGE_SYMPTOMS      Profile -> Symptom -> Retrieval -> Risk
query -> Planner ----|                         -> Recommendation -> Hospital -> Final
         Agent       +--> RETRIEVE_KNOWLEDGE   Retrieval -> grounded answer
                     +--> COMPARE_CONDITIONS   retrieval per side -> contrast
                     +--> MEDICATION_SAFETY    deterministic refusal
                     +--> OUT_OF_SCOPE         deterministic refusal
```

## Trained components and measured results

Three supervised models are trained from the datasets in `data/` and evaluated
on held-out data: the operation-selection planner, a triage classifier (used by
the Risk Agent, with the red-flag keyword screen kept as a guardrail) and a
MedQuAD question-intent classifier (used by the Retrieval Agent). Operation
selection is compared against a fixed pipeline, a rule policy and Gemini as a
planner; retrieval and the full pipeline are evaluated too.

See [TRAINING_AND_EVALUATION.md](TRAINING_AND_EVALUATION.md) for the datasets,
protocols, commands and the measured numbers. Raw metrics live in `results/`
and are displayed on the app's Evaluation page.

## Safety

This is an academic prototype. It does not diagnose, prescribe, or provide real-time
hospital availability. Use verified medical sources and qualified clinicians for real care.
