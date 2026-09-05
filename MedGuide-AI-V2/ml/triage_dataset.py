"""Builds the triage dataset used to train the Triage & Symptom Analysis agent.

Each seed vignette is written in patient language and labelled with one of four
urgency tiers, following Emergency Severity Index (ESI) style criteria:

    EMERGENCY  ESI 1-2  red-flag presentation, needs care immediately
    HIGH       ESI 3    needs same-day / urgent assessment
    MODERATE   ESI 4    should see a doctor soon, not urgent
    LOW        ESI 5    self-limiting, self-care is reasonable

Seeds are expanded with neutral phrasing wrappers (wrappers never add duration
or severity words, so they cannot change the correct label). The train/test
split is done over *seeds*, not over the expanded rows, so no phrasing of a
test vignette is ever seen during training.
"""

import csv
from pathlib import Path

OUT_FILE = Path(__file__).resolve().parent.parent / "data" / "triage_dataset.csv"

SEEDS = [
    # ---------------- EMERGENCY (ESI 1-2) ----------------
    ("crushing chest pain spreading to my left arm and jaw", "EMERGENCY"),
    ("chest pain and sweating, feels like pressure on my chest", "EMERGENCY"),
    ("i cannot breathe properly and my lips look blue", "EMERGENCY"),
    ("severe difficulty breathing, i can only speak a few words", "EMERGENCY"),
    ("coughing up blood since this morning", "EMERGENCY"),
    ("vomiting blood and feeling faint", "EMERGENCY"),
    ("sudden weakness on one side of my body and slurred speech", "EMERGENCY"),
    ("my face has drooped on one side and my arm is numb", "EMERGENCY"),
    ("i had a seizure a few minutes ago and i am confused", "EMERGENCY"),
    ("my child is unresponsive and floppy", "EMERGENCY"),
    ("heavy bleeding from a deep cut that will not stop", "EMERGENCY"),
    ("worst headache of my life, started suddenly like a thunderclap", "EMERGENCY"),
    ("i took an overdose of tablets", "EMERGENCY"),
    ("i am having thoughts of ending my life", "EMERGENCY"),
    ("severe abdominal pain with a rigid hard stomach", "EMERGENCY"),
    ("high fever with a stiff neck and i cannot tolerate light", "EMERGENCY"),
    ("swollen tongue and throat closing after eating peanuts", "EMERGENCY"),
    ("whole body rash with swelling of the face and trouble breathing", "EMERGENCY"),
    ("fainted and hit my head, vomiting since then", "EMERGENCY"),
    ("i am pregnant with heavy bleeding and severe belly pain", "EMERGENCY"),
    ("burns over my arm with blistering skin from boiling water", "EMERGENCY"),
    ("cannot pass urine at all since yesterday and severe lower belly pain", "EMERGENCY"),
    ("sudden loss of vision in one eye", "EMERGENCY"),
    ("chest tightness with cold sweat and vomiting", "EMERGENCY"),
    ("my newborn baby is not breathing normally and is grunting", "EMERGENCY"),
    ("snake bite on my leg an hour ago", "EMERGENCY"),
    ("severe electric shock and my heart is racing", "EMERGENCY"),
    ("confused, drowsy and cannot be woken properly", "EMERGENCY"),
    ("severe allergic reaction with hives all over and wheezing", "EMERGENCY"),
    ("chest pain that started while resting and is not going away", "EMERGENCY"),
    ("i am short of breath at rest and cannot lie flat", "EMERGENCY"),
    ("large amount of blood in my stool just now", "EMERGENCY"),
    ("my child swallowed a button battery", "EMERGENCY"),
    ("i drank a cleaning liquid by mistake", "EMERGENCY"),
    ("severe head injury after a road accident", "EMERGENCY"),
    ("my arm is broken and the bone is visible", "EMERGENCY"),
    ("i feel my heart racing very fast and i almost fainted", "EMERGENCY"),
    ("sudden severe pain between my shoulder blades tearing in nature", "EMERGENCY"),
    ("fever with fits in my three year old right now", "EMERGENCY"),
    ("i cannot move my legs since this morning", "EMERGENCY"),

    # ---------------- HIGH (ESI 3) ----------------
    ("increased wheezing and shortness of breath, i am asthmatic", "HIGH"),
    ("shortness of breath when i walk up stairs, started this week", "HIGH"),
    ("fever of 103 for four days with shivering", "HIGH"),
    ("i am 71 and have had fever and weakness for three days", "HIGH"),
    ("my two year old has fever and is refusing all fluids", "HIGH"),
    ("vomiting everything i drink since yesterday, feeling very dry", "HIGH"),
    ("watery diarrhoea fifteen times today and dizzy on standing", "HIGH"),
    ("i am diabetic and my foot wound is swollen with pus", "HIGH"),
    ("blood in my urine with burning and back pain", "HIGH"),
    ("black stools for two days and i feel lightheaded", "HIGH"),
    ("severe one sided abdominal pain moving to my groin", "HIGH"),
    ("painful swollen red calf, it feels warm", "HIGH"),
    ("fever with a productive cough and pain on breathing in", "HIGH"),
    ("my blood sugar reading is above 400 and i feel very thirsty", "HIGH"),
    ("palpitations with dizziness on and off today", "HIGH"),
    ("i am pregnant and have not felt the baby move since morning", "HIGH"),
    ("eye is red and painful with blurred vision", "HIGH"),
    ("spreading red skin around a wound with fever", "HIGH"),
    ("severe testicular pain that started a few hours ago", "HIGH"),
    ("i am on blood thinners and fell down the stairs", "HIGH"),
    ("severe ear pain with discharge and fever", "HIGH"),
    ("i am 68 and my ankles have swollen and i get breathless lying down", "HIGH"),
    ("unable to keep any food down after chemotherapy", "HIGH"),
    ("sudden severe back pain and i cannot pass urine properly", "HIGH"),
    ("high fever with a rash that does not fade when pressed", "HIGH"),
    ("i am 80 and became more confused than usual today", "HIGH"),
    ("bad headache with vomiting and my vision is doubled", "HIGH"),
    ("infant of six weeks with fever", "HIGH"),
    ("i am asthmatic and my inhaler is not helping much today", "HIGH"),
    ("fever for five days and now i feel breathless when walking", "HIGH"),
    ("i am 74 with a bad cough and my oxygen reading is low", "HIGH"),
    ("severe pain and swelling in my hand after a dog bite yesterday", "HIGH"),
    ("jaundice with yellow eyes and dark urine since two days", "HIGH"),
    ("i am pregnant with a bad headache and swollen hands", "HIGH"),
    ("my wound from surgery is opening and leaking fluid", "HIGH"),
    ("severe vomiting for a day and i have not passed urine since morning", "HIGH"),
    ("i have kidney disease and my legs have swollen suddenly", "HIGH"),
    ("a lump in my breast that has grown quickly this month", "HIGH"),
    ("i am 66 and had a brief episode where i could not speak", "HIGH"),
    ("my blood pressure reading is 190 over 110 and i feel unwell", "HIGH"),

    # ---------------- MODERATE (ESI 4) ----------------
    ("dry cough that has lasted six days and is not improving", "MODERATE"),
    ("fever, cough, body ache and fatigue for three days", "MODERATE"),
    ("cold, fever and headache for seven days", "MODERATE"),
    ("sore throat with difficulty swallowing solid food for four days", "MODERATE"),
    ("headache almost every day for two weeks", "MODERATE"),
    ("stomach pain and loose motions for four days", "MODERATE"),
    ("burning while passing urine for three days", "MODERATE"),
    ("itchy rash on both arms spreading slowly over a week", "MODERATE"),
    ("lower back pain for ten days after lifting weights", "MODERATE"),
    ("knee pain and swelling for a week after a fall", "MODERATE"),
    ("ear blocked with mild pain for five days", "MODERATE"),
    ("acidity and chest burning after meals for two weeks", "MODERATE"),
    ("dizziness on standing up for the past week", "MODERATE"),
    ("hair fall increasing for the last three months", "MODERATE"),
    ("mouth ulcers that keep coming back for a month", "MODERATE"),
    ("toothache for four days that worsens at night", "MODERATE"),
    ("irregular periods for the last four months", "MODERATE"),
    ("cannot sleep properly for three weeks and feel anxious", "MODERATE"),
    ("weight loss of four kilos over two months without trying", "MODERATE"),
    ("swollen gums that bleed when brushing for two weeks", "MODERATE"),
    ("persistent tiredness for a month even after resting", "MODERATE"),
    ("recurring pimples with painful boils for six weeks", "MODERATE"),
    ("joint pain in fingers every morning for a month", "MODERATE"),
    ("blurred vision when reading for the last two weeks", "MODERATE"),
    ("constipation for eight days with bloating", "MODERATE"),
    ("mild fever coming and going for five days", "MODERATE"),
    ("dandruff with an itchy scalp for two months", "MODERATE"),
    ("numbness and tingling in my fingers for three weeks", "MODERATE"),
    ("cough with white phlegm for eight days, no fever now", "MODERATE"),
    ("ankle pain and mild swelling for six days after twisting it", "MODERATE"),
    ("white patches in my mouth for two weeks", "MODERATE"),
    ("frequent urination at night for the last three weeks", "MODERATE"),
    ("itchy scaly patches on my elbows for a month", "MODERATE"),
    ("shoulder pain that limits lifting my arm for two weeks", "MODERATE"),
    ("stomach bloating and gas almost daily for three weeks", "MODERATE"),
    ("low mood and loss of interest for the past month", "MODERATE"),
    ("ringing in both ears for two weeks", "MODERATE"),
    ("heel pain when i walk in the morning for three weeks", "MODERATE"),
    ("nosebleeds twice a week for the last month", "MODERATE"),
    ("cracked and peeling skin on my feet for six weeks", "MODERATE"),

    # ---------------- LOW (ESI 5) ----------------
    ("slight fever and a bit of tiredness since this morning", "LOW"),
    ("runny nose and sneezing since last night", "LOW"),
    ("mild sore throat since yesterday", "LOW"),
    ("mild headache today after skipping lunch", "LOW"),
    ("one loose motion this morning, otherwise fine", "LOW"),
    ("small mosquito bite that is a little itchy", "LOW"),
    ("mild body ache after playing football yesterday", "LOW"),
    ("slight stomach discomfort after eating spicy food", "LOW"),
    ("dry lips and mild throat irritation today", "LOW"),
    ("mild sneezing when i clean dusty rooms", "LOW"),
    ("a single pimple on my chin", "LOW"),
    ("slight burning eyes after long screen time today", "LOW"),
    ("mild neck stiffness after sleeping in a bad position", "LOW"),
    ("small paper cut on my finger", "LOW"),
    ("hiccups on and off this afternoon", "LOW"),
    ("mild gas and burping since lunch", "LOW"),
    ("one bruise on my knee after bumping a table", "LOW"),
    ("slight cough for one day, no fever", "LOW"),
    ("mild soreness in my shoulder after gym today", "LOW"),
    ("i feel slightly tired today after a late night", "LOW"),
    ("mild itching on my scalp today", "LOW"),
    ("small blister on my heel from new shoes", "LOW"),
    ("light period cramps today", "LOW"),
    ("mild nasal block in the morning only", "LOW"),
    ("slightly watery eyes since morning", "LOW"),
    ("mild ringing in my ear for a few minutes today", "LOW"),
    ("minor sunburn on my nose from this afternoon", "LOW"),
    ("mild toe pain after stubbing it just now", "LOW"),
    ("slight sneezing and itchy nose this morning only", "LOW"),
    ("mild acidity after a heavy dinner last night", "LOW"),
    ("tired eyes after studying all evening", "LOW"),
    ("small red patch where my watch strap rubs", "LOW"),
    ("mild jaw ache after chewing gum today", "LOW"),
    ("slight dizziness once when i stood up quickly today", "LOW"),
    ("mild leg cramp last night while sleeping", "LOW"),
    ("itchy skin after wearing a new woollen sweater today", "LOW"),
    ("mild throat dryness after singing loudly yesterday", "LOW"),
    ("one small mouth ulcer since yesterday", "LOW"),
    ("slight wrist ache after typing all day", "LOW"),
    ("mild forehead warmth but thermometer shows normal", "LOW"),
]

# Wrappers only change phrasing. They never add duration, age or severity
# information, so the label of a seed is preserved by construction.
WRAPPERS = [
    "{}",
    "i have {}",
    "doctor, {}",
    "{}. what should i do",
    "{}. please advise",
    "hi, {} and i am worried",
    "hello doctor {}",
    "i am experiencing {}",
]


def build():
    rows = []
    for seed_id, (text, label) in enumerate(SEEDS):
        for wrapper in WRAPPERS:
            rows.append({
                "seed_id": seed_id,
                "text": wrapper.format(text),
                "label": label,
            })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed_id", "text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    return rows


if __name__ == "__main__":
    rows = build()
    counts = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(f"Wrote {len(rows)} vignettes from {len(SEEDS)} seeds to {OUT_FILE}")
    for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label:10s} {count}")
