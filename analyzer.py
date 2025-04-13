# === File: analyzer.py ===
import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import List, Dict

# Load environment variables
load_dotenv()

# Initialize models
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and metadata
faiss_index = faiss.read_index("faiss_index/vector.index")
with open("faiss_index/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Initialize psychoanalytic profile
psycho_profile = {
    "psychoanalysis": {},
    "about_user": {}
}

# === Emotion Analysis ===
def analyze_emotions(user_input: str) -> Dict[str, float]:
    transformer_output = emotion_classifier(user_input)
    emotions = {item['label']: round(item['score'], 4) for item in transformer_output[0] if item['score'] > 0.01}

    with open("emotions.txt", "w", encoding="utf-8") as f:
        json.dump(emotions, f, indent=2)

    return emotions

# === RAG Retrieval ===
def retrieve_context(query: str, k: int = 5) -> List[str]:
    query_vec = embedding_model.encode([query])[0].astype("float32")
    if query_vec.shape[0] != faiss_index.d:
        raise ValueError(f"Query vector dimension ({query_vec.shape[0]}) does not match FAISS index dimension ({faiss_index.d})")

    _, indices = faiss_index.search(np.array([query_vec]), k)
    retrieved = [metadata[i].get("text", "") for i in indices[0] if i < len(metadata)]

    with open("rag_output.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(retrieved):
            f.write(f"Chunk {i + 1}:\n{chunk}\n\n")

    return retrieved

# === Update Psychoanalytic Profile ===
def update_psycho_profile(user_input: str, context_chunks: List[str]) -> None:
    prompt = f"""
You are a psychoanalyst AI. Based on the user input and context, provide:
1. Good/Bad thinking patterns
2. Axioms or core beliefs
3. Cognitive distortions
4. Defense mechanisms
5. Maladaptive patterns
6. Inferred beliefs / self-schema
7. Emotional regulation patterns

Input:
{user_input}

Context:
{''.join(context_chunks[:3])}
"""

    # Simulate psychoanalytic inference
    psychoanalysis_output = {
        "thinking_patterns": {"User demonstrates proactive optimism.": {"short_term": 0.7, "long_term": 0.6}},
        "axioms": {"Believes effort leads to success.": {"short_term": 0.8, "long_term": 0.7}},
        "cognitive_distortions": {"Possible minimization of potential risks.": {"short_term": 0.3, "long_term": 0.2}},
        "defense_mechanisms": {"Might intellectualize emotions when excited.": {"short_term": 0.4, "long_term": 0.3}},
        "maladaptive_patterns": {"Mild dependency on external validation for emotional uplift.": {"short_term": 0.35, "long_term": 0.25}},
        "inferred_beliefs": {"Views emotional connection as part of personal success.": {"short_term": 0.5, "long_term": 0.4}},
        "emotional_regulation": {"Utilizes positive feedback loops to maintain mood.": {"short_term": 0.6, "long_term": 0.5}}
    }

    psycho_profile["psychoanalysis"] = psychoanalysis_output

    with open("psychoanalysis.txt", "w", encoding="utf-8") as f:
        json.dump(psycho_profile["psychoanalysis"], f, indent=2)

        # Simulate about_user inference (you can replace with a real model later)
    about_user_output = {
        "certain": {
            "The user values joy and friendship.": True,
            "The user feels confident in their abilities.": True
        },
        "unsure": {
            "The user may have recently formed new social connections.": {
                "short_term": 0.7,
                "long_term": 0.5
            },
            "The user may derive motivation from social energy.": {
                "short_term": 0.6,
                "long_term": 0.4
            }
        }
    }

    psycho_profile["about_user"] = about_user_output

    with open("about_user.txt", "w", encoding="utf-8") as f:
        json.dump(psycho_profile["about_user"], f, indent=2)


# === Expose profile for external use ===
def get_psycho_profile() -> Dict[str, Dict]:
    return psycho_profile
