# Training and evaluation

Every number below was produced by running the scripts in `ml/` on this machine.
The raw outputs are saved as JSON in `results/` and shown on the app's
**Evaluation** page. Nothing here is estimated or copied from literature.

## 1. Where the medical knowledge comes from

| Corpus | File | Size | Used for |
| --- | --- | --- | --- |
| MedQuAD | `data/medquad.csv` | 16,407 usable Q&A pairs (NIH: GARD, GHR, NIDDK, NINDS, CancerGov, NHLBI, CDC, MedlinePlus) | TF-IDF retrieval of reference medical information; source of the question-intent labels |
| MedDialog (English) | `data/meddialog/english-train.json` | patient/doctor conversations | retrieval of similar real consultations |

The language model (Gemini) is **not fine-tuned**. Medical knowledge is acquired
by *retrieval* (RAG): the retrieved passages are inserted into the prompt for
the single generation call. Three components *are* trained with supervised
learning, and all three are evaluated below.

## 2. Trained component 1 — operation-selection planner

This is the research question of the project: given a free-text query, **which
operation should the framework run?** The orchestrator is not a fixed sequence;
the planner (`agents/planner.py`) picks one of six operations, each of which
expands into its own agent chain in `workflow.py`.

| Operation | Runs | Why it exists |
| --- | --- | --- |
| `EMERGENCY_ESCALATE` | nothing else — immediate escalation | red-flag wording must not wait for retrieval |
| `TRIAGE_SYMPTOMS` | symptoms → retrieval → risk → recommendation → hospital → answer | the patient describes their own complaint |
| `RETRIEVE_KNOWLEDGE` | retrieval → grounded answer | a general medical information question |
| `COMPARE_CONDITIONS` | retrieval of *each* side → contrast | one passage cannot answer a two-sided question |
| `MEDICATION_SAFETY` | deterministic refusal | naming drugs or doses is out of bounds |
| `OUT_OF_SCOPE` | deterministic refusal | it is not a general-purpose chatbot |

- Dataset: `python -m ml.plan_dataset` → `data/plan_dataset.csv`, 3,040 queries.
  Knowledge and comparison queries are built from real MedQuAD questions and
  topic names; symptom and emergency queries reuse the triage seeds and their
  8 phrasings; medication and out-of-scope queries are authored.
- Script: `python -m ml.train_planner` → word TF-IDF (1–2) ∪ char TF-IDF (3–5)
  + logistic regression (`class_weight="balanced"`).
- Split: `GroupShuffleSplit` over source groups (712 train / 178 test groups,
  2,457 / 583 examples), so no rephrasing of a test query is seen in training.
- Grouped 5-fold CV accuracy: **0.918 ± 0.014**.

Five policies were then compared on the identical held-out split
(`python -m ml.eval_planner` → `results/planner_eval.json`, 583 queries):

| Policy | Accuracy | Macro F1 | Emergencies not bypassed | Prescriptions not refused | Mean decision time |
| --- | --- | --- | --- | --- | --- |
| Fixed pipeline (the original single-path system) | 0.220 | 0.060 | 1.000 | 1.000 | <0.1 ms |
| Hand-written rule policy | 0.842 | 0.820 | 0.727 | 0.000 | 0.02 ms |
| Trained planner alone | 0.837 | 0.815 | 0.784 | 0.000 | 2.1 ms |
| **Deployed policy** (trained planner + safety overrides) | **0.863** | **0.851** | 0.614 | 0.000 | 2.0 ms |
| Gemini as planner (89/90 sampled queries answered) | 0.876 | 0.884 | 0.000 | 0.000 | 6.96 s |

Per-operation F1 for the deployed policy: comparison 1.00, knowledge 1.00,
medication 1.00, out-of-scope 0.89, symptoms 0.75, emergency 0.47.

Two results deserve care rather than celebration:

1. **The fixed pipeline is the honest baseline and it loses badly** (0.220).
   That is the measurable argument for choosing operations rather than running a
   fixed chain: the same framework, the same corpora, only the decision layer
   differs.
2. **The planner's emergency recall is poor (0.39)**, because an emergency query
   phrased calmly looks like an ordinary symptom report. Reporting only the
   planner would therefore overstate safety, and reporting the 0.614 figure as a
   patient-facing failure rate would understate it: a query routed to
   `TRIAGE_SYMPTOMS` still meets the red-flag screen and the trained triage
   classifier one stage later. Measuring the *pipeline* instead of the planner
   (`system_safety` in the same JSON) gives, over the 88 emergency queries in
   the split: 34 escalated by the planner bypass, 54 escalated by the triage
   stage, **0 missed — escalation rate 1.000**. Safety here is a property of
   defence in depth, not of the classifier.

Gemini plans slightly better than the trained model but is ~3,500× slower per
decision and depends on API availability (1 of 90 calls returned nothing during
the run), so it is evaluated as a baseline and not deployed as the router.

## 3. Trained component 2 — triage classifier

- Script: `python -m ml.train_triage`
- Data: `data/triage_dataset.csv`, built by `ml/triage_dataset.py` from 160
  labelled ESI-style symptom vignettes, each expanded with 8 neutral phrasings
  → 1,280 examples, 4 balanced classes (EMERGENCY / HIGH / MODERATE / LOW).
- Features: word TF-IDF (1–2 grams) ∪ character TF-IDF (3–5 grams).
- Classifier: calibrated linear SVM.
- Split: `GroupShuffleSplit` **over seed vignettes**, 1,024 train / 256 test, so
  no phrasing of a test vignette is ever seen during training.
- Decision rule: argmax, but escalated to EMERGENCY when
  P(EMERGENCY) ≥ 0.2. The threshold was tuned on a validation split taken from
  the training data only, minimising under-triage.

| System (256 held-out examples) | Accuracy | Macro F1 | Under-triage rate |
| --- | --- | --- | --- |
| Existing hand-written rule ladder (baseline) | 0.219 | 0.198 | 0.600 |
| Trained classifier, plain argmax | 0.688 | 0.677 | 0.500 |
| Trained classifier, safety-tuned threshold | 0.688 | 0.600 | 0.100 |
| **Deployed hybrid** (classifier + red-flag keyword screen) | **0.688** | 0.600 | **0.100** |

5-fold grouped cross-validation accuracy: **0.712 ± 0.051**.

*Under-triage rate* = share of EMERGENCY cases assigned a lower level; it is the
safety metric that matters most, and the trained system reduces it from 0.60 to
0.10. The safety-tuned threshold trades macro F1 for that reduction, because
some HIGH cases are deliberately escalated to EMERGENCY.

Honest limitations: the vignettes are author-written, not clinician-validated,
and the HIGH/EMERGENCY boundary is intrinsically fuzzy. The numbers describe
*this* dataset, not clinical performance.

## 4. Trained component 3 — question-intent classifier

- Script: `python -m ml.train_intent`
- Data: `data/intent_dataset.csv`, 6,841 MedQuAD questions labelled with the
  intent implied by the question template (symptoms, treatment, causes,
  prevention, diagnosis, genetics, frequency, prognosis, definition). Disease
  names are removed so the model learns the intent, not the disease.
- Model: TF-IDF (1–2 grams) + logistic regression, stratified 80/20 split
  (5,472 train / 1,369 test).
- Held-out accuracy and macro F1: **1.000**; 5-fold CV accuracy 0.999 ± 0.001.

This near-perfect score is expected and must be reported as such: MedQuAD
questions are generated from fixed templates, so intent recognition is an easy
task. It is *not* diagnostic accuracy. The classifier is used in
`agents/medical_agent.py` to expand the retrieval query with intent-specific
terms.

## 5. Retrieval evaluation (RAG quality)

- Script: `python -m ml.eval_retrieval`
- 300 randomly sampled MedQuAD questions (seed 42) used as queries against all
  16,407 documents.

| Protocol | R@1 | R@3 | R@5 | MRR@10 | Topic R@3 |
| --- | --- | --- | --- | --- | --- |
| As deployed (index includes the question text) | 0.293 | 0.520 | 0.680 | 0.453 | 0.907 |
| Answer-only index (harder, honest setting) | 0.240 | 0.467 | 0.657 | 0.398 | 0.877 |

Mean retrieval latency 10.2 ms (p95 10.5 ms). Exact-document recall is modest
because many MedQuAD entries about the same disease are near-duplicates; the
metric that matters for the prompt is **topic recall@3 (0.91)** — the right
disease topic is retrieved for 9 of 10 queries.

## 6. End-to-end pipeline evaluation

- Script: `python -m ml.eval_pipeline` (add `--llm` to include real Gemini
  generation; the default run uses the offline fallback so it is free and
  deterministic).
- 32 held-out triage seeds, full seven-agent pipeline.

| Metric | Value |
| --- | --- |
| Triage accuracy (final level) | 0.719 |
| Critical safety violation rate (missed EMERGENCY) | 0.000 |
| Immediate-care advice on EMERGENCY cases | 1.000 |
| Answer structure compliance | 0.719 |
| Groundedness (answer sentences overlapping retrieved context) | 0.33 |
| Mean end-to-end latency, LLM excluded | 0.069 s (p95 0.05 s, max 1.18 s) |

Slowest agent: Medical Knowledge Retrieval (0.064 s mean); every other
non-LLM agent is below 11 ms. Structure compliance is measured against the
five-section symptom template, so escalated cases — which use the shorter
emergency template on purpose — count against it. With Gemini enabled the total is dominated by the
single API call.

Groundedness here is a coarse lexical proxy (≥2 shared content words per
sentence), not a human judgement, and is reported as such.

## 7. Answering questions taken straight from MedQuAD

A query that is a general medical question ("What causes Urinary Incontinence
?") is not a symptom report, so it skips triage and is answered from the
matching corpus row instead (`agents/query_router.py` →
`agents/knowledge_agent.py`). Ranking is helped by three corrections in
`agents/medical_agent.py`: a candidate whose question repeats ≥80% of the words
of the asked question is treated as the same question and wins outright (TF-IDF
alone ranked "How to prevent Kidney Dysplasia" above "How to prevent Kidney
Disease"), candidates whose wording matches the asked intent are promoted
within the same disease topic, and rows that contain only lists of external
links are dropped when explanatory text is available.

- Script: `python -m ml.eval_knowledge_qa`
- 200 randomly sampled MedQuAD questions (seed 42) replayed through the full
  pipeline, Gemini disabled so the reply is the corpus extract itself.

| Metric | Plain TF-IDF | As deployed |
| --- | --- | --- |
| Source row ranked first | 0.385 | **0.905** |
| Source row in top 3 | 0.680 | **0.920** |
| Correct disease topic first | 0.920 | **0.955** |
| Routed as a knowledge question | – | 0.990 |
| Reply overlaps the source answer (≥15% of its words) | – | 0.960 |
| Mean latency, LLM excluded | – | 0.019 s |

So for a question copied out of the dataset, the answer is drawn from that
question's own MedQuAD row about 9 times in 10, and from the right disease
topic 19 times in 20. When nothing relevant is retrieved the app says so rather
than producing generic text. Emergency wording still bypasses this route.
The patient-facing answer never names the corpus or the source; the app shows
the matched question, source and similarity separately in the UI.

## 8. Reproducing everything

```bash
pip install -r requirements.txt
python -m ml.intent_dataset      # build data/intent_dataset.csv
python -m ml.triage_dataset      # build data/triage_dataset.csv
python -m ml.plan_dataset        # build data/plan_dataset.csv
python -m ml.train_intent        # -> models/intent_classifier.joblib, results/intent_metrics.json
python -m ml.train_triage        # -> models/triage_classifier.joblib, results/triage_metrics.json
python -m ml.train_planner       # -> models/planner_classifier.joblib, results/planner_metrics.json
python -m ml.eval_planner        # -> results/planner_eval.json (uses Gemini if a key is set)
python -m ml.eval_retrieval      # -> results/retrieval_metrics.json
python -m ml.eval_pipeline       # -> results/pipeline_metrics.json
python -m ml.eval_knowledge_qa   # -> results/knowledge_qa_metrics.json
python -m pytest tests -q        # 25 tests
```

All scripts are seeded (`random_state=42`) and re-runnable end to end.
