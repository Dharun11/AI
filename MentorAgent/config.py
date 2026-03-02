import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY", "")

# LLM_PROVIDER can be "gemini" or "openai"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o") # User asked for gpt 5.1, but using 4o as default
