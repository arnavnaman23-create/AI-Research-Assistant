
import os
from groq import Groq

MODEL_NAME = "openai/gpt-oss-120b"

def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured.")
    return Groq(api_key=api_key)
