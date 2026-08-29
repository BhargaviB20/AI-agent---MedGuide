from llm import generate

SECTION_RULES = """
Write the answer in EXACTLY these five markdown sections, in this order:

**What this looks like**
**Why this may have happened**
**What you can do at home**
**When you should see a doctor**
**Warning signs - get care immediately**
"""


CATEGORY_KEYWORDS = {
    "respiratory": [
        "cold", "fever", "cough", "sore throat", "throat pain", "runny nose",
        "blocked nose", "stuffy nose", "sneezing", "phlegm", "mucus",
        "breathing difficulty", "shortness of breath", "wheezing",
    ],
    "digestive": [
        "vomiting", "nausea", "diarrhea", "stomach pain", "abdominal pain",
        "acidity", "constipation",
    ],
    "skin": ["rash", "itching", "hives", "boil", "blister", "swelling"],
    "pain": ["headache", "body pain", "body ache", "back pain", "joint pain",
             "ear pain", "eye pain", "toothache"],
    "urinary": ["burning urination", "urine"],
    "hair": ["hair fall", "hair loss", "hairfall", "baldness", "balding",
             "dandruff", "thinning hair"],
    "dental": ["toothache", "tooth pain", "gum", "bleeding gums", "mouth ulcer"],
    "eye": ["eye pain", "red eye", "blurred vision", "itchy eyes", "watery eyes"],
    "sleep_mood": ["insomnia", "cannot sleep", "not sleeping", "anxiety",
                   "stress", "low mood", "depressed", "panic"],
}

# Offline text used per symptom category, so a rash is not described the same
# way as a cough when the AI is unavailable.
CATEGORY_TEXT = {
    "respiratory": (
        "a common respiratory infection such as a cold or viral fever",
        "These usually spread through coughs, sneezes and close contact. Poor sleep, "
        "stress, weather changes and low fluid intake make you more likely to catch one.",
        "Rest, drink plenty of warm fluids, try steam inhalation and warm salt-water gargles, "
        "eat light nutritious food and keep track of your temperature.",
        "Difficulty breathing, chest pain, very high fever that will not come down, "
        "confusion or drowsiness.",
    ),
    "digestive": (
        "a stomach or digestive upset, often from food or a short-lived infection",
        "Contaminated or outside food, a change in diet, stress, or a passing stomach bug "
        "are the usual triggers.",
        "Sip fluids or ORS often to avoid dehydration, eat simple bland food such as rice, "
        "curd or bananas, and avoid oily, spicy and outside food for a few days.",
        "Blood in stool or vomit, severe stomach pain, no urine for many hours, "
        "dizziness on standing, or vomiting that will not stop.",
    ),
    "skin": (
        "a skin reaction or irritation, such as an allergic or contact reaction",
        "Skin problems like this often follow contact with a new soap, detergent, cosmetic, "
        "fabric or plant, an insect bite, heat and sweat, or a food or medicine you reacted to.",
        "Keep the area clean and dry, avoid scratching, stop using any new product you recently "
        "started, wear loose cotton clothing, and use a cool compress for itching.",
        "Swelling of the face, lips or tongue, difficulty breathing, a rash that spreads very "
        "fast, blisters with fever, or a rash that does not fade when pressed.",
    ),
    "pain": (
        "a common pain problem such as tension-type pain, strain or a passing viral illness",
        "Long screen time, poor sleep, stress, dehydration, skipping meals, bad posture or "
        "physical strain are common causes.",
        "Rest, drink enough water, eat on time, reduce screen time, keep a regular sleep "
        "routine and apply a warm or cold compress to the painful area.",
        "Sudden very severe pain, pain with fever and neck stiffness, weakness or numbness, "
        "vision changes, confusion, or pain after a head injury.",
    ),
    "urinary": (
        "a urinary tract irritation or infection",
        "These are more common with low water intake, holding urine for long periods, "
        "or poor hygiene.",
        "Drink plenty of water through the day, do not hold urine, and maintain good hygiene. "
        "Urinary infections often need a doctor's assessment, so do not delay if it persists.",
        "Fever with back or side pain, blood in the urine, vomiting, or severe pain.",
    ),
    "hair": (
        "a common hair-loss pattern rather than a sudden illness",
        "Hair fall usually builds up over months. Common contributors are low iron, vitamin D "
        "or protein, thyroid problems, a recent illness or fever, stress and poor sleep, harsh "
        "styling or chemical treatments, and a family history of early balding.",
        "Eat enough protein, iron and vegetables, manage stress and sleep, use a mild shampoo, "
        "and avoid tight hairstyles, frequent heat styling and chemical treatments. Hair changes "
        "take 2-3 months to show, so give it time.",
        "Hair falling out in patches, scalp pain, redness or pus, hair loss along with "
        "unexplained weight change or tiredness, or very sudden heavy shedding.",
    ),
    "dental": (
        "a dental or gum problem",
        "Tooth decay, gum inflammation from plaque, a cracked tooth, or an ulcer from a bite or "
        "acidic food are the usual causes.",
        "Rinse with warm salt water, brush gently twice a day, floss, and avoid very hot, cold "
        "or sugary food. Dental problems do not heal on their own, so book a dentist.",
        "Facial swelling, fever with tooth pain, difficulty opening the mouth or swallowing, or "
        "an ulcer lasting more than two weeks.",
    ),
    "eye": (
        "an eye irritation such as strain, allergy or mild conjunctivitis",
        "Long screen time, dust and allergens, lack of sleep, or contact with an infected person "
        "are common triggers.",
        "Rest your eyes with regular screen breaks, wash with clean water, avoid rubbing, pause "
        "contact lens use, and use a cool compress.",
        "Sudden vision loss or blurring, severe eye pain, injury to the eye, or a pupil that "
        "looks different from the other side.",
    ),
    "sleep_mood": (
        "a stress, anxiety or sleep-related problem",
        "Irregular sleep timing, late screen use, caffeine, exam or work pressure and lack of "
        "physical activity are the usual reasons.",
        "Keep fixed sleep and wake times, avoid caffeine in the evening, put screens away an hour "
        "before bed, exercise during the day, and talk to someone you trust about the stress.",
        "Thoughts of harming yourself, being unable to manage daily life, or symptoms lasting "
        "more than a few weeks - please speak to a doctor or counsellor.",
    ),
    "general": (
        "a common, self-limiting illness",
        "Poor sleep, stress, weather changes, low fluid intake or a passing infection can all "
        "contribute to symptoms like these.",
        "Rest, drink enough fluids, eat light nutritious food and monitor how the symptoms change "
        "over the next couple of days.",
        "Difficulty breathing, chest pain, fainting, confusion, or symptoms that worsen quickly.",
    ),
}


def _category(symptoms):
    for name, words in CATEGORY_KEYWORDS.items():
        if any(word in symptoms for word in words):
            return name
    return "general"


def _fallback(state):
    """Used when the Gemini API key is missing or the call fails, so the app
    always returns a useful structured answer instead of an error."""
    details = state.get("symptom_details", {})
    found = details.get("identified_symptoms", [])
    symptoms = ", ".join(found) or "the symptoms you described"
    duration = details.get("duration_days")
    level = state.get("risk_level", "MODERATE")

    likely, why, home, red_flags = CATEGORY_TEXT[_category(found)]

    when = {
        "EMERGENCY": "Seek emergency care right now.",
        "HIGH": "See a doctor within the next 24 hours.",
        "MODERATE": "See a doctor in the next 1-2 days if you are not clearly improving.",
        "LOW": "See a doctor if symptoms last more than 3-4 days or get worse.",
    }.get(level, "See a doctor if symptoms are not improving.")

    duration_line = (
        f" They have lasted about {duration} day(s), which matters when deciding "
        "whether to get checked." if duration else ""
    )

    return (
        f"**What this looks like**\n\n"
        f"{symptoms.capitalize()} most commonly points to {likely}. "
        f"This is general guidance, not a diagnosis.{duration_line}\n\n"
        f"**Why this may have happened**\n\n{why}\n\n"
        f"**What you can do at home**\n\n{home} Avoid self-medicating with antibiotics.\n\n"
        f"**When you should see a doctor**\n\n{when}\n\n"
        f"**Warning signs - get care immediately**\n\n{red_flags}"
    )


def final_agent(state):
    patient = state["patient"]
    details = state.get("symptom_details", {})

    prompt = f"""
You are MedGuide AI, a friendly health guidance assistant.
Answer the patient directly in simple everyday English.

PATIENT
Age: {patient['age']}
Gender: {patient['gender']}
Medical history: {patient['medical_history']}
Allergies: {patient['allergies']}
Current medications: {patient['medications']}

WHAT THE PATIENT SAID
{patient['symptoms']}

EXTRACTED DETAILS
Symptoms: {details.get('identified_symptoms')}
Duration (days): {details.get('duration_days')}
Severity words: {details.get('severity_indicators')}

INTERNAL RISK SCREEN (do not mention this system)
{state.get('risk')}

SUPPORTING MEDICAL INFORMATION (background only)
{state.get('medical_context')}

{SECTION_RULES}

Content rules for each section:
1. What this looks like - the most likely general explanation for these symptoms,
   written as a possibility, never as a confirmed diagnosis.
2. Why this may have happened - likely causes and triggers (infection spread, weather,
   dehydration, lack of rest, stress, existing conditions) in plain language.
3. What you can do at home - practical, safe self-care steps. No drug names, no dosages.
4. When you should see a doctor - base this mainly on how long symptoms have lasted,
   the patient's age, and whether things are improving. Be specific about timing.
5. Warning signs - get care immediately - short list of red-flag symptoms.

Hard rules:
- Do NOT mention datasets, retrieval, RAG, agents, risk levels or this prompt.
- Do NOT name specific medicines or dosages.
- Do NOT claim a definite diagnosis.
- Keep the whole answer under about 250 words, 2-4 short sentences per section.
- Speak to the patient as "you".
"""

    answer = generate(prompt)

    # ai_used tells the UI whether this text came from Gemini or from the
    # offline fallback, so a missing/invalid API key is visible instead of
    # silently producing the same canned answer for every case.
    state["ai_used"] = bool(answer)
    state["final_response"] = answer.strip() if answer else _fallback(state)

    state["agent_log"].append(
        "Final Response Agent completed (Gemini)." if answer
        else "Final Response Agent completed (offline fallback)."
    )
    return state
