import json
from datetime import datetime
from pathlib import Path

import streamlit as st

import database
import llm
from workflow import run_medguide

# =========================================================
# DATABASE
# =========================================================

database.init_db()


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MedGuide AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CSS
# =========================================================
# NOTE: every HTML block below is written WITHOUT leading indentation.
# Streamlit's markdown treats 4-space indented lines as a code block, which is
# why the raw <div> tags were showing up on screen before.

st.markdown(
    """
<style>
.stApp { background: #06111f; color: #eef4ff; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1450px; }

section[data-testid="stSidebar"] {
background: linear-gradient(180deg, #071526 0%, #06111f 100%);
border-right: 1px solid #19324b;
}
section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }

.brand { display: flex; align-items: center; gap: 12px; padding: 8px 8px 18px 8px; }
.brand-icon {
width: 44px; height: 44px; border-radius: 12px;
display: flex; align-items: center; justify-content: center;
background: linear-gradient(135deg, #1ea7ff, #5ce1e6);
color: #04101c; font-size: 25px; font-weight: 800;
}
.brand-title { font-size: 22px; font-weight: 800; line-height: 1.1; }
.brand-subtitle { color: #9db1c8; font-size: 12px; margin-top: 3px; }

.tip-card {
margin-top: 22px; padding: 18px; border: 1px solid #21445d; border-radius: 14px;
background: linear-gradient(145deg, #0b2637, #081827);
}
.tip-title { color: #4dd7ff; font-weight: 700; font-size: 16px; }
.tip-text { color: #c6d5e5; line-height: 1.6; font-size: 13px; margin-top: 10px; }

.topbar { border-bottom: 1px solid #19324b; padding-bottom: 18px; margin-bottom: 18px; }
.topbar-title { font-size: 30px; font-weight: 800; letter-spacing: -0.5px; }
.topbar-subtitle { color: #9db1c8; margin-top: 5px; }

.disclaimer {
padding: 12px 16px; border: 1px solid #4d3b1d; background: #211b10;
color: #e7d7b4; border-radius: 10px; font-size: 12px; margin-bottom: 20px;
}

div[data-testid="stForm"] {
background: #0b1929; border: 1px solid #1c3852; border-radius: 16px; padding: 22px;
}
.stTextInput input, .stTextArea textarea, .stNumberInput input {
background: #081525 !important; color: #edf5ff !important;
border: 1px solid #28445d !important; border-radius: 10px !important;
}
div[data-baseweb="select"] > div { background: #081525 !important; border-color: #28445d !important; }

.stButton > button, .stFormSubmitButton > button {
border-radius: 10px; border: 1px solid #24527a; background: #0d2137;
color: #eaf5ff; font-weight: 650;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
border-color: #38bdf8; color: white;
}

.user-card {
background: linear-gradient(135deg, #172f59, #102444);
border: 1px solid #31578a; border-radius: 14px; padding: 18px 20px;
margin: 20px 0; color: #e5efff;
}
.user-question { margin-top: 10px; font-size: 16px; line-height: 1.6; }

div[data-testid="stVerticalBlockBorderWrapper"]:has(.answer-title) {
border: 1px solid #24527a !important; border-left: 4px solid #38bdf8 !important;
border-radius: 16px; background: linear-gradient(145deg, #101f31, #0a1726);
padding: 6px 24px 16px 24px; margin-top: 18px;
}
.answer-title { font-size: 23px; font-weight: 750; margin: 10px 0 6px 0; color: #eef5ff; }

.metric-card {
background: #0b1929; border: 1px solid #203b55; border-radius: 14px;
padding: 15px; text-align: center;
}
.metric-value { font-size: 25px; font-weight: 800; }
.metric-label { color: #9eb2c8; font-size: 12px; }

.badge {
display: inline-block; padding: 5px 12px; border-radius: 999px;
font-size: 12px; font-weight: 800;
}
.badge-low { background: #123b29; color: #63e58b; }
.badge-moderate { background: #493817; color: #ffd166; }
.badge-high { background: #492717; color: #ffae72; }
.badge-emergency { background: #4a1b24; color: #ff6b7a; }
.badge-info { background: #16334a; color: #7cc4ff; }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_risk_badge(risk_text):
    text = str(risk_text).upper()

    if "EMERGENCY" in text:
        return "EMERGENCY", "badge-emergency"
    if "INFO" in text:
        return "INFO", "badge-info"
    if "HIGH" in text:
        return "HIGH", "badge-high"
    if "MODERATE" in text:
        return "MODERATE", "badge-moderate"
    return "LOW", "badge-low"


def build_report(patient, result):
    return f"""# MedGuide AI Consultation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Patient

Age: {patient['age']}
Gender: {patient['gender']}
Location: {patient['location']}
Medical history: {patient['medical_history']}
Allergies: {patient['allergies']}
Medications: {patient['medications']}

## Symptoms

{patient['symptoms']}

## AI Guidance

{result.get('final_response', '')}

## Suggested Care Pathway

{result.get('recommendation', '')}

{result.get('hospital_navigation', '')}

## Response Time

{result.get('response_time_seconds', '-')} seconds

---

AI-assisted health guidance. This is not a medical diagnosis.
"""


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
<div class="brand">
<div class="brand-icon">✚</div>
<div>
<div class="brand-title">MedGuide AI</div>
<div class="brand-subtitle">Your AI Medical Assistant</div>
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["💬 Chat", "👤 Patient Profile", "📜 History", "📊 Evaluation", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown(
        """
<div class="tip-card">
<div class="tip-title">♥ Health Tips</div>
<div class="tip-text">
Stay hydrated, eat balanced meals, get enough sleep, and exercise regularly
for a healthier lifestyle.
</div>
</div>
""",
        unsafe_allow_html=True,
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    """
<div class="topbar">
<div class="topbar-title">Personalized Healthcare Guidance</div>
<div class="topbar-subtitle">
AI-assisted health guidance based on your symptoms and medical information.
</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="disclaimer">
⚠️ <b>Educational prototype only.</b>
MedGuide AI does not diagnose, prescribe, or replace a qualified healthcare
professional. For emergency symptoms, seek immediate professional care.
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# CHAT PAGE
# =========================================================

if page == "💬 Chat":

    with st.form("patient_form", clear_on_submit=False):
        left, right = st.columns(2)

        with left:
            age = st.number_input("Age", min_value=0, max_value=120, value=25)
            gender = st.selectbox("Gender", ["Female", "Male", "Other", "Prefer not to say"])
            history = st.text_area("Medical history", "None reported", height=90)
            allergies = st.text_area("Allergies", "None reported", height=90)

        with right:
            medications = st.text_area("Current medications", "None reported", height=90)
            location = st.text_input("Location", "Chennai")
            symptoms = st.text_area(
                "Medical question / symptoms",
                height=170,
                placeholder="Example: I have cold, fever and headache for 7 days. What should I do?",
            )

        submitted = st.form_submit_button("🚀 Get Guidance", use_container_width=True)

    if submitted:
        if not symptoms.strip():
            st.error("Please enter your symptoms or question.")
        else:
            patient = {
                "age": age,
                "gender": gender,
                "medical_history": history,
                "allergies": allergies,
                "medications": medications,
                "location": location,
                "symptoms": symptoms,
            }

            status = st.empty()

            def on_step(step_name, index, total):
                status.caption(f"Step {index}/{total}: {step_name}")

            with st.spinner("Analyzing your symptoms..."):
                result = run_medguide(patient, on_step=on_step)

            status.empty()

            database.save_consultation(
                patient,
                result.get("risk", ""),
                result.get("final_response", ""),
                result.get("response_time_seconds", 0),
            )

            st.markdown(
                f"""
<div class="user-card">
<b>👤 Your question</b>
<div class="user-question">{symptoms}</div>
</div>
""",
                unsafe_allow_html=True,
            )

            risk, badge_class = get_risk_badge(result.get("risk", ""))

            if result.get("risk_level") != "EMERGENCY" and not result.get("ai_used"):
                st.warning(
                    "AI not connected - showing the built-in offline answer. "
                    f"Reason: {llm.LAST_ERROR or 'unknown'}"
                )

            # The AI answer is real markdown, so it is rendered with st.markdown
            # inside a bordered container instead of being pushed into an HTML
            # div (that broke the formatting and showed raw tags).
            with st.container(border=True):
                st.markdown(
                    f'<div class="answer-title">🩺 MedGuide AI '
                    f'<span class="badge {badge_class}">{risk}</span></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(result.get("final_response", "Unable to generate a response."))

                if result.get("query_type") != "knowledge_question":
                    st.markdown("**Suggested care pathway**")
                    st.write(result.get("recommendation", ""))
                    st.write(result.get("hospital_navigation", ""))

            st.caption(
                "AI-assisted health guidance. This does not replace professional medical advice."
            )

            st.download_button(
                "⬇️ Download consultation report",
                build_report(patient, result),
                file_name=f"medguide_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            )


# =========================================================
# PATIENT PROFILE
# =========================================================

elif page == "👤 Patient Profile":
    st.markdown("## 👤 Patient Profile")
    st.info("Enter patient information in the Chat page before running an analysis.")


# =========================================================
# HISTORY
# =========================================================

elif page == "📜 History":
    st.markdown("## 📜 Consultation History")

    records = database.get_history()

    if not records:
        st.info("No consultations yet.")
    else:
        if st.button("🗑️ Clear history"):
            database.clear_history()
            st.rerun()

        for record in records:
            risk, badge_class = get_risk_badge(record["risk_level"])

            with st.expander(f"{record['timestamp']} • Age {record['age']} • {risk}"):
                st.markdown(
                    f'<span class="badge {badge_class}">{risk}</span>',
                    unsafe_allow_html=True,
                )
                st.write("**Location:**", record["location"])
                st.write("**Symptoms:**", record["symptoms"])
                st.markdown(record["final_response"])
                st.caption(f"Response time: {record['response_time_seconds']} seconds")


# =========================================================
# EVALUATION
# =========================================================

elif page == "📊 Evaluation":
    st.markdown("## 📊 Evaluation")

    st.markdown("### Measured results")
    st.caption(
        "Produced by the scripts in `ml/`: train_triage.py, train_intent.py, "
        "eval_retrieval.py, eval_pipeline.py and eval_knowledge_qa.py. "
        "Re-run them to refresh these numbers."
    )

    RESULTS_DIR = Path(__file__).resolve().parent / "results"
    RESULT_FILES = {
        "Triage classifier vs rules": "triage_metrics.json",
        "Question-intent classifier": "intent_metrics.json",
        "MedQuAD retrieval": "retrieval_metrics.json",
        "End-to-end pipeline": "pipeline_metrics.json",
        "MedQuAD question answering": "knowledge_qa_metrics.json",
    }

    found_any = False
    for title, filename in RESULT_FILES.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            continue
        found_any = True
        with st.expander(f"{title}  ({filename})"):
            data = json.loads(path.read_text(encoding="utf-8"))
            data.pop("cases_detail", None)
            st.json(data, expanded=False)

    if not found_any:
        st.info(
            "No measured results yet. Run `python -m ml.train_triage`, "
            "`python -m ml.train_intent`, `python -m ml.eval_retrieval` and "
            "`python -m ml.eval_pipeline` first."
        )

    st.markdown("### Live smoke test")
    st.caption("Runs the full pipeline on a few predefined cases.")

    TEST_CASES = [
        {
            "name": "Mild fever",
            "symptoms": "Slight fever and a bit of tiredness since this morning.",
            "expected": "LOW",
        },
        {
            "name": "Persistent cough",
            "symptoms": "Dry cough that has lasted 6 days and is not improving.",
            "expected": "MODERATE",
        },
        {
            "name": "Breathing difficulty",
            "symptoms": "Severe difficulty breathing and chest tightness.",
            "expected": "EMERGENCY",
        },
        {
            "name": "Multiple symptoms",
            "symptoms": "Fever, cough, body ache and fatigue for 3 days.",
            "expected": "MODERATE",
        },
        {
            "name": "Asthma symptoms",
            "symptoms": "Increased wheezing and shortness of breath, I have asthma.",
            "expected": "HIGH",
        },
    ]

    st.dataframe(TEST_CASES, use_container_width=True, hide_index=True)

    if st.button("▶️ Run all test cases", use_container_width=True):
        base_patient = {
            "age": 30,
            "gender": "Female",
            "medical_history": "None",
            "allergies": "None",
            "medications": "None",
            "location": "Chennai",
        }

        rows = []
        correct = 0

        with st.spinner("Running evaluation..."):
            for case in TEST_CASES:
                patient = {**base_patient, "symptoms": case["symptoms"]}
                result = run_medguide(patient)

                actual_level = result.get("risk_level", "UNKNOWN")
                passed = actual_level == case["expected"]
                correct += int(passed)

                rows.append(
                    {
                        "Test Case": case["name"],
                        "Expected": case["expected"],
                        "Actual": actual_level,
                        "Status": "PASS" if passed else "FAIL",
                        "Response Time (s)": result.get("response_time_seconds", "-"),
                    }
                )

        accuracy = round((correct / len(TEST_CASES)) * 100, 1)

        times = [float(r["Response Time (s)"]) for r in rows if r["Response Time (s)"] != "-"]
        average_time = sum(times) / len(times) if times else 0

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{accuracy}%</div>'
                f'<div class="metric-label">Accuracy</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{correct}/{len(TEST_CASES)}</div>'
                f'<div class="metric-label">Correct Cases</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{average_time:.2f}s</div>'
                f'<div class="metric-label">Average Response Time</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("### Results")
        st.dataframe(rows, use_container_width=True, hide_index=True)


# =========================================================
# ABOUT
# =========================================================

else:
    st.markdown("## ℹ️ About MedGuide AI")

    st.markdown(
        """
**MedGuide AI** is an academic healthcare guidance prototype.

It combines the patient's profile and symptoms with medical knowledge retrieved
from the **MedQuAD** and **MedDialog** datasets, and uses a pretrained language
model to generate one simple, patient-friendly answer that explains what the
symptoms may be, why they may have happened, what can be done at home, and when
a doctor should be consulted.

The retrieved medical data is used internally to support the answer and is not
displayed to the user. Triage combines a trained classifier with a deterministic
red-flag keyword screen that can escalate a case to EMERGENCY but never lower it,
and only the final explanation uses the language model.

Measured results for every component are on the Evaluation page.

This system is for academic demonstration only and is not a replacement for
professional medical diagnosis or treatment.
"""
    )
