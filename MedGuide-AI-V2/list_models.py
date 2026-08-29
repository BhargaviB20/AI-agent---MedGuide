"""Prints the Gemini models your API key can actually use.

Run:  python list_models.py
Then put a working name in .env as GEMINI_MODEL=...
"""
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for model in client.models.list():
    actions = getattr(model, "supported_actions", None) or []
    if not actions or "generateContent" in actions:
        print(model.name)
