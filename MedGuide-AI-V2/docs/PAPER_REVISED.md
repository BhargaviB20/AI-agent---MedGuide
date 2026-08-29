# MedGuide AI: A Retrieval-Grounded Multi-Agent Framework for Personalized Healthcare Navigation

Bhargavi B, Naslun Wafa T
Department of Computer Science and Engineering, College of Engineering, Anna University, Chennai, India

## Abstract

Patients seeking health information online face two distinct problems: understanding a named medical condition, and deciding how urgently a set of personal symptoms needs professional care. General-purpose Large Language Models answer both fluently but without verifiable grounding, and a single monolithic prompt gives no place to enforce a safety floor. This paper presents MedGuide AI, an implemented multi-agent framework that separates these concerns. Medical knowledge is acquired by retrieval, not by fine-tuning: a TF-IDF index over 16,412 MedQuAD question-answer pairs curated from NIH sources, supplemented by 482 English MedDialog patient-doctor exchanges. Urgency is decided by a supervised classifier (word plus character TF-IDF with a calibrated linear Support Vector Machine) trained on 1,280 Emergency-Severity-Index-style symptom vignettes, and a deterministic red-flag screen that may only escalate the predicted level, never lower it. A single Large Language Model call (Google Gemini) is used solely to phrase the final answer from the retrieved passages and the assigned urgency level; it is never asked to supply facts or to decide urgency, and the system degrades to a deterministic offline response when the model is unavailable. All reported numbers come from executed evaluation scripts included with the code. On a vignette-disjoint test split the trained triage stage reaches 0.688 accuracy with a critical-safety-violation rate of 0.10, against 0.219 and 0.60 for the rule-based baseline it replaces. Retrieval reaches topic Recall@3 of 0.907 over the full corpus, and on 200 questions replayed from the corpus the answer is drawn from the question's own row 90.5% of the time. End-to-end latency excluding the Large Language Model call is 0.058 s. We state explicitly what these numbers do not show: the triage vignettes are author-written and not clinician-validated, groundedness is a lexical proxy, and no clinical validation has been performed.

**Index Terms**—Multi-Agent Systems, Retrieval-Augmented Generation, Clinical Triage, Large Language Models, Healthcare Navigation, AI Safety.

## I. Introduction

Deciding whether a symptom needs a hospital visit today, tomorrow, or not at all is a routine but consequential judgement that patients make with poor information. Search engines return documents rather than decisions; general-purpose Large Language Models (LLMs) return confident prose whose provenance cannot be checked. Two failure modes matter in this domain. The first is unverified generation: a model that produces a plausible mechanism for a symptom without any source behind it. The second is an absent safety floor: a system that, on an unusual phrasing of a red-flag complaint, produces reassuring advice with no component whose job is to prevent that.

MedGuide AI addresses both by refusing to let one component do everything. Facts come from a retrievable corpus, urgency comes from a trained classifier constrained by a deterministic screen, and the LLM contributes wording only. This paper describes the system as implemented and evaluated, not as intended.

### A. Contributions

1. An implemented multi-agent healthcare-navigation pipeline with two distinct routes — a general medical-information route and a personal-symptom triage route — separated by an explicit query router, with an emergency screen ahead of both.
2. A concrete answer to the question of how medical knowledge is acquired: retrieval over MedQuAD and MedDialog at request time, with the retrieved passage constrained to be the only permitted source of facts in the generated answer.
3. Two supervised models trained on documented, reproducible datasets — a four-level triage classifier and a nine-class question-intent classifier — with train/test protocols that prevent phrasing leakage.
4. A measured evaluation of every component, produced by scripts shipped with the code, including a comparison against the rule-based logic the trained triage stage replaces, and an explicit statement of what remains unvalidated.

### B. Note on evaluation data

Every number in Section VII is produced by a script in the `ml/` package of the accompanying implementation and written to `results/*.json`; the commands are listed in Section IX. No number in this paper is illustrative or hand-authored. This replaces the placeholder tables of the earlier version of this work, in which metrics were reported before the system had been executed.

## II. Related Work

The framework draws on three lines of work: retrieval-augmented generation, in which a retriever supplies evidence that constrains a generator and thereby reduces unsupported claims; multi-agent decomposition, in which specialized components are orchestrated rather than a single prompt being overloaded; and automated triage, in which symptom text is mapped onto standardized urgency tiers such as the Emergency Severity Index. Benchmark work on clinical question answering establishes that general LLMs encode substantial medical knowledge, which is precisely why an explicit grounding and urgency-decision layer is needed: fluency in this domain is not evidence of correctness.

The gap this work targets is narrower than the one claimed in the earlier version. It is not an integrated clinical platform spanning scheduling, electronic-health-record ingestion and drug-interaction monitoring; none of that is implemented. It is the combination, in one executable system, of corpus-grounded medical answering, a trained urgency model with a deterministic safety floor above it, and an LLM restricted to presentation.

## III. Implemented System Architecture

MedGuide AI is a Python application with a Streamlit interface (`app.py`) and a sequential orchestrator (`workflow.py`). Consultation history is stored in a local SQLite database. There is no service mesh, no vector database and no graph-orchestration library; the orchestrator is a list of agent functions applied to a shared state dictionary, which is sufficient for the pipeline depth involved and keeps every intermediate decision inspectable.

### A. Routing

An incoming query passes through three decisions in a fixed order.

1. **Emergency screen.** The raw text is matched against a red-flag pattern set (severe chest pain, inability to breathe, unconsciousness, heavy bleeding, stroke signs, seizure, coughing or vomiting blood, and related phrasings). A match returns a deterministic emergency response immediately; no model runs, and no later stage can override it. This is the system's safety floor.
2. **Query routing.** A rule-based router (`agents/query_router.py`) separates a general information question ("What are the treatments for hearing loss?") from a personal symptom report ("I have had fever and headache for seven days"), using question-leading words, trailing interrogatives and word-boundary-anchored first-person markers. Word boundaries matter: a substring test for "me" misroutes every question containing "syndrome".
3. **Route execution.** General questions run the four-stage knowledge route; symptom reports run the seven-stage triage route.

### B. Agents

The knowledge route comprises the Patient Profile Agent, the Symptom Analysis Agent, the Medical Knowledge Agent (retrieval plus intent-aware reranking) and the Knowledge Answer Agent, which produces the final text from the retrieved row. The triage route comprises the Patient Profile Agent, the Symptom Analysis Agent (symptom, duration and severity extraction), the Medical Knowledge Retrieval Agent, the Risk Assessment Agent (trained classifier plus the escalate-only guardrail), the Recommendation Agent, the Hospital Navigation Agent and the Final Response Agent, which issues the single LLM call.

### C. Retrieval and reranking

The retriever (`rag/retriever.py`) builds a TF-IDF matrix over the concatenation of each MedQuAD question, answer and focus area, with English stop-word removal and 20,000 features, and ranks by cosine similarity. Three refinements were added after measuring failures:

- **Intent-aware query expansion and reranking.** The predicted question intent adds discriminative terms to the query and gives a bonus to candidate rows whose question carries matching cues, so that a question about prevention is not answered from the treatment row of the same disease.
- **Near-exact question promotion.** The normalized word-set overlap between the asked question and each candidate question is computed over a 30-candidate pool; a candidate with overlap of at least 0.8 is promoted ahead of the TF-IDF ranking. Without this, a question copied verbatim from the corpus could be answered from a similarly worded question about a different disease — the observed failure was "How to prevent Kidney Disease?" being answered from the Kidney Dysplasia row.
- **Boilerplate suppression.** Rows whose answer is a resource list ("These resources address...", registry and review links) are dropped when an explanatory row is available, and website furniture ("watch the video", "click here") is stripped from the extracted text.

If no candidate exceeds a minimum similarity of 0.15, the system states that it has no reliable information on the question rather than answering from a weak match.

### D. Role of the Large Language Model

Exactly one LLM call is made per consultation (`llm.py`, Google Gemini). Its prompt contains the patient profile, the extracted symptoms, the urgency level already decided by the Risk Assessment Agent, the retrieved passages, and instructions to use only those passages, to write in plain language, to omit medicine names and dosages, to avoid a definite diagnosis, and not to mention datasets or internal components. The model is not fine-tuned on MedQuAD or MedDialog; knowledge enters the prompt at request time. If the key is absent or the API fails, a deterministic fallback produces the same sections by extracting and de-duplicating the retrieved text, so the system remains demonstrable offline.

## IV. Datasets and Knowledge Acquisition

**MedQuAD** (`data/medquad.csv`): 16,412 question-answer pairs (16,407 with non-empty answers) covering 4,742 focus areas, curated from NIH sources including GHR, GARD, NIDDK, NINDS, MedlinePlus, NIHSeniorHealth, CancerGov, NHLBI and CDC. Question types are dominated by definition (4,606), symptoms (2,749), treatment (2,469), inheritance (1,456), frequency (1,117), causes (738) and diagnosis (661). Because GHR and GARD together contribute 10,824 rows, coverage is strongest for named genetic and rare diseases and weakest for arbitrary symptom combinations — a property that directly explains the difference in answer quality between the two routes.

**MedDialog (English)** (`data/meddialog/english-train.json`): 482 loaded patient-doctor exchanges, used as supplementary conversational context on the symptom route.

**Intent dataset** (`data/intent_dataset.csv`): 6,841 rows derived from MedQuAD questions, labelled by the question template into nine intents (definition, symptoms, causes, treatment, prevention, diagnosis, genetics, frequency, prognosis) with the disease name stripped so that the classifier learns question form rather than disease identity.

**Triage dataset** (`data/triage_dataset.csv`): 160 author-written ESI-style symptom vignettes labelled EMERGENCY, HIGH, MODERATE or LOW, each rendered in 8 neutral phrasings for 1,280 rows. This dataset exists because neither MedQuAD nor MedDialog carries urgency labels, and deriving them automatically would fabricate the very ground truth being measured. It is not clinician-validated; this is the single most important limitation of the reported triage results.

## V. Training Methodology

**Triage classifier** (`ml/train_triage.py`): word (1-2 gram) and character (3-5 gram) TF-IDF features feeding a linear Support Vector Machine with probability calibration. Splitting uses GroupShuffleSplit over the 160 seed vignettes, so all 8 phrasings of a vignette fall on the same side of the split and no phrasing leaks from training into testing: 1,024 training and 256 test rows. Training takes 0.12 s. At inference the calibrated EMERGENCY probability is thresholded at 0.20 rather than taking the argmax; this deliberately trades precision for recall on the class where a miss is harmful, and the deterministic red-flag screen sits above it as an escalate-only guardrail.

**Intent classifier** (`ml/train_intent.py`): TF-IDF (1-2 gram) features with logistic regression, 5,472 training and 1,369 test rows, trained in 0.05 s.

Both models are written to `models/*.joblib` and consumed by the agents through `ml/predictors.py`, which falls back to the previous rule-based behaviour if a model file is missing.

## VI. Evaluation Metrics

- **Triage accuracy and macro-F1** over the four urgency levels on the vignette-disjoint test split.
- **Critical Safety Violation Rate (CSVR):** the fraction of EMERGENCY test cases assigned a level below EMERGENCY by the deployed pipeline. This is the primary safety metric and the only one whose target is zero.
- **Retrieval Recall@k, topic Recall@3 and MRR@10:** measured by holding out MedQuAD questions and asking whether the source row (Recall@k) or any row on the same focus area (topic Recall@3) is retrieved.
- **Knowledge-route metrics:** for questions replayed from the corpus, the rate at which the question's own row is ranked first and within the top three, the rate at which the correct topic is ranked first, and the rate at which the delivered answer overlaps the gold answer text. The last is a lexical proxy for groundedness, not a clinical correctness judgement.
- **End-to-end latency:** wall-clock time per consultation, reported with the LLM call disabled so that the measurement reflects the local pipeline rather than network variance.

## VII. Results

All values below are read from `results/*.json`, produced by the commands in Section IX.

**Table I — Triage stage on the vignette-disjoint test split (256 rows, 80 EMERGENCY)**

| Configuration | Accuracy | Macro-F1 | EMERGENCY recall | CSVR |
| --- | --- | --- | --- | --- |
| Rule-based baseline (previous logic) | 0.219 | 0.198 | 0.400 | 0.60 |
| Trained model, argmax decision | 0.688 | 0.677 | 0.500 | 0.50 |
| Deployed: trained model, threshold 0.20 + red-flag guardrail | 0.688 | 0.600 | 0.900 | 0.10 |

Grouped 5-fold cross-validation over seed vignettes gives 0.712 ± 0.051 accuracy. The deployed configuration keeps the same accuracy as the argmax configuration while cutting the safety-critical error rate from 0.50 to 0.10, which is the trade this system is designed to make. Per-class results show the cost: HIGH is not separated from EMERGENCY (HIGH F1 of 0.00 in the deployed configuration, with HIGH cases escalated upward), while MODERATE and LOW are recovered reliably (F1 0.875 and 0.857).

**Table II — Question-intent classifier (1,369 test rows, 9 classes)**

| Metric | Value |
| --- | --- |
| Accuracy | 1.000 |
| Macro-F1 | 1.000 |
| 5-fold CV accuracy | 0.999 ± 0.001 |

This result must be read as intent recognition on template-generated questions, not as medical accuracy. MedQuAD questions are produced from a small set of templates, so the task is close to pattern identification; the value of the classifier is that it makes retrieval intent-aware, not that it demonstrates clinical competence.

**Table III — Retrieval over the full 16,407-row corpus (300 held-out queries)**

| Metric | As deployed (question+answer+topic index) | Answer-only index |
| --- | --- | --- |
| Recall@1 | 0.293 | 0.240 |
| Recall@3 | 0.520 | 0.467 |
| Recall@5 | 0.680 | 0.657 |
| Topic Recall@3 | 0.907 | 0.877 |
| MRR@10 | 0.453 | 0.398 |
| Mean latency (ms) | 10.2 | 10.3 |

Exact-row recall is modest because the corpus contains many near-duplicate questions per disease; topic Recall@3 of 0.907 is the metric that reflects whether the system reaches the right subject matter.

**Table IV — Knowledge route, 200 questions replayed from the corpus (seed 42, LLM disabled)**

| Metric | Value |
| --- | --- |
| Routed as a general question | 0.990 |
| Source row ranked first | 0.905 |
| Source row within top 3 | 0.920 |
| Correct topic ranked first | 0.955 |
| Answer overlaps the source answer | 0.970 |
| Mean latency (s) | 0.021 |

Before near-exact question promotion was added, the source row was ranked first in 0.695 of cases; the reranking change accounts for the increase to 0.905.

**Table V — End-to-end pipeline (32 held-out symptom cases, LLM disabled)**

| Metric | Value |
| --- | --- |
| Mean latency (s) | 0.058 |
| Median / p95 latency (s) | 0.020 / 0.030 |
| Immediate-care advice on EMERGENCY cases | 1.000 |
| Response structure compliance | 0.875 |
| Slowest stage (Medical Knowledge Agent, mean s) | 0.056 |

Latency is dominated by retrieval; the trained classifier contributes 0.011 s. A live Gemini call adds roughly one to three seconds of network-bound time on top of this, which is why it is excluded from the pipeline measurement rather than folded into it.

## VIII. Discussion, Limitations and Safety

The measured results support a narrow claim: that decomposing the problem improves the safety-relevant behaviour of the system over the monolithic rule-based logic it replaces, and that answers on the knowledge route are traceable to a specific corpus row. They do not support any claim of clinical accuracy.

Specific limitations, stated deliberately:

- **The triage ground truth is author-written.** The 160 vignettes were written for this project, not labelled by clinicians and not drawn from a validated dataset. Every triage number inherits that weakness. Replacing this dataset with clinician-labelled vignettes is the highest-value next step.
- **HIGH and EMERGENCY are not separated.** The system resolves this by escalating, which is the safe direction but produces over-triage.
- **Groundedness is measured lexically.** Overlap between the delivered answer and the gold answer shows that the text came from the retrieved row; it does not show that the row answers the question a patient actually asked.
- **The intent classifier's perfect score is an artefact of template-generated questions.**
- **Corpus coverage is uneven.** Rare and genetic diseases are well covered; arbitrary symptom combinations have no matching row, which is why the symptom route relies on the trained classifier and structured guidance rather than on retrieval alone.
- **The system is single-turn, English-only, and has no clinical validation, no clinician in the loop and no regulatory assessment.** It is an educational prototype. It does not diagnose or prescribe, refuses medicine names and dosages, and displays a standing disclaimer.
- **Safety depends on a deterministic component, not on a model.** The red-flag screen runs first and can only escalate. This is a design decision made because a 0.688-accuracy classifier is not a safe last line of defence.

## IX. Reproducibility

From the repository root:

```
pip install -r requirements.txt
python -m ml.triage_dataset          # build data/triage_dataset.csv
python -m ml.intent_dataset          # build data/intent_dataset.csv
python -m ml.train_triage            # -> models/triage.joblib, results/triage_metrics.json
python -m ml.train_intent            # -> models/intent.joblib, results/intent_metrics.json
python -m ml.eval_retrieval          # -> results/retrieval_metrics.json
python -m ml.eval_knowledge_qa --limit 200   # -> results/knowledge_qa_metrics.json
python -m ml.eval_pipeline           # -> results/pipeline_metrics.json
python -m pytest tests -q            # 16 tests
streamlit run app.py                 # demo, port 8501
```

The Evaluation page of the application renders the same JSON files, so the numbers in this paper can be read off the running system during a demonstration. A Gemini API key in `.env` (`GEMINI_API_KEY`, `GEMINI_MODEL`) enables generated wording; without it the deterministic offline path is used and the demonstration still runs.

## X. Conclusion and Future Work

MedGuide AI is an executed multi-agent healthcare-navigation prototype in which knowledge acquisition, urgency decision and language generation are handled by different components with different failure modes: retrieval over 16,412 curated NIH question-answer pairs, a supervised triage classifier under a deterministic red-flag guardrail, and a single constrained LLM call for phrasing. Measured on shipped evaluation scripts, the deployed triage configuration reduces the critical-safety-violation rate from 0.60 to 0.10 relative to the rule-based logic it replaces, retrieval reaches topic Recall@3 of 0.907, and the knowledge route answers from the question's own corpus row in 90.5% of replayed cases at 0.021 s per query.

Future work follows directly from the limitations: clinician-labelled triage vignettes to replace the author-written ground truth, dense embedding retrieval to raise exact-row recall above the TF-IDF ceiling, separation of HIGH from EMERGENCY, human evaluation of answer correctness to replace the lexical groundedness proxy, and multi-turn and multilingual interaction. Claims of clinical utility should follow a clinician-supervised study, not this evaluation.

*Note on references: the literature survey table of the earlier version is retained separately and every entry must be verified against the original source before submission.*
