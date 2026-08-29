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
the single generation call. Two components *are* trained with supervised
learning, and both are evaluated below.

## 2. Trained component 1 — triage classifier

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

## 3. Trained component 2 — question-intent classifier

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

## 4. Retrieval evaluation (RAG quality)

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

## 5. End-to-end pipeline evaluation

- Script: `python -m ml.eval_pipeline` (add `--llm` to include real Gemini
  generation; the default run uses the offline fallback so it is free and
  deterministic).
- 32 held-out triage seeds, full seven-agent pipeline.

| Metric | Value |
| --- | --- |
| Triage accuracy (final level) | 0.688 |
| Critical safety violation rate (missed EMERGENCY) | 0.100 |
| Immediate-care advice on EMERGENCY cases | 1.000 |
| Answer structure compliance | 0.875 |
| Groundedness (answer sentences overlapping retrieved context) | 0.40 |
| Mean end-to-end latency, LLM excluded | 0.058 s (p95 0.03 s, max 1.20 s) |

Slowest agent: Medical Knowledge Retrieval (0.056 s mean); every other
non-LLM agent is below 11 ms. With Gemini enabled the total is dominated by the
single API call.

Groundedness here is a coarse lexical proxy (≥2 shared content words per
sentence), not a human judgement, and is reported as such.

## 6. Reproducing everything

```bash
pip install -r requirements.txt
python -m ml.intent_dataset      # build data/intent_dataset.csv
python -m ml.triage_dataset      # build data/triage_dataset.csv
python -m ml.train_intent        # -> models/intent_classifier.joblib, results/intent_metrics.json
python -m ml.train_triage        # -> models/triage_classifier.joblib, results/triage_metrics.json
python -m ml.eval_retrieval      # -> results/retrieval_metrics.json
python -m ml.eval_pipeline       # -> results/pipeline_metrics.json
python -m pytest tests -q
```

All scripts are seeded (`random_state=42`) and re-runnable end to end.
