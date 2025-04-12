# === File: responder.py ===
import openai
from dotenv import load_dotenv
import os
import json
from typing import List, Dict

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(user_text: str,
                      psycho_profile: Dict[str, Dict],
                      retrieved_chunks: List[str],
                      model: str = "gpt-4o") -> str:
    """
    Generate a thoughtful response using GPT-4o, considering the psychoanalytic profile and retrieved memory.
    """

    prompt = f"""
You are a deeply compassionate, emotionally intelligent AI therapist.
Here is the user's message:
"{user_text}"

Here are psychoanalytic observations about the user:
{json.dumps(psycho_profile.get("psychoanalysis", {}), indent=2)}

Here is relevant context retrieved from therapy literature:
{''.join(retrieved_chunks[:3])}

Please respond like a therapist would — warm, curious, open-minded, and reflective. Use their language when possible.
"""

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an insightful and compassionate AI therapist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()
