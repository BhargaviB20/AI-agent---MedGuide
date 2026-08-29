import os

from dotenv import load_dotenv

load_dotenv()

# Reason the last call failed, shown in the UI so a missing or invalid key is
# obvious instead of silently falling back to canned text.
LAST_ERROR = ""

PLACEHOLDER_KEYS = {"", "your_gemini_api_key", "your_key_here", "changeme"}

# Tried in order if GEMINI_MODEL is unset or that model is no longer available.
MODEL_CANDIDATES = [
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash-001",
    "gemini-1.5-flash",
]


def generate(prompt: str) -> str:
    """Call the Gemini API. Returns '' on any failure so agents can
    fall back to their rule-based logic instead of crashing the app."""
    global LAST_ERROR

    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()

    if api_key.lower() in PLACEHOLDER_KEYS:
        LAST_ERROR = (
            "GEMINI_API_KEY is missing or still set to a placeholder. "
            "Put your real key in the .env file and restart the app."
        )
        print(f"[llm.generate] {LAST_ERROR}")
        return ""

    try:
        from google import genai
    except ImportError:
        LAST_ERROR = "google-genai is not installed. Run: pip install google-genai"
        print(f"[llm.generate] {LAST_ERROR}")
        return ""

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        LAST_ERROR = f"{type(e).__name__}: {e}"
        print(f"[llm.generate] Could not create the Gemini client: {LAST_ERROR}")
        return ""

    # Google retires model names over time, so try the configured model first
    # and then known working alternatives instead of failing outright.
    candidates = []
    for name in [os.getenv("GEMINI_MODEL"), *MODEL_CANDIDATES]:
        if name and name not in candidates:
            candidates.append(name)

    for model_name in candidates:
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            text = (response.text or "").strip()

            if not text:
                LAST_ERROR = (
                    f"{model_name} returned an empty response "
                    "(possibly blocked by safety filters)."
                )
                print(f"[llm.generate] {LAST_ERROR}")
                continue

            if model_name != candidates[0]:
                print(f"[llm.generate] Using fallback model: {model_name}")

            LAST_ERROR = ""
            return text

        except Exception as e:
            LAST_ERROR = f"{model_name} -> {type(e).__name__}: {e}"
            print(f"[llm.generate] {LAST_ERROR}")

    print("[llm.generate] All Gemini models failed, using offline fallback.")
    return ""
