from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import google.generativeai as genai
import os

# ======================
# CONFIG
# ======================
genai.configure(api_key="AQ.Ab8RN6KZqzirCxcSB6OtC0YBpylv4LtW4sPzxYVCV2udUOo2dA")  # ← Paste your key here

MODEL_NAME = "gemini-3.6-flash"
model_ai = genai.GenerativeModel(MODEL_NAME)

# Load your skill classifier
skill_model = joblib.load("skill_classifier.pkl")
FEATURE_NAMES = ["accuracy", "reaction_time", "hesitation", "retries"]

app = FastAPI(title="NurtureBloom AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ======================
# Request Body
# ======================
class GameStats(BaseModel):
    accuracy: float = Field(..., ge=0, le=100)
    reaction_time: float = Field(..., ge=0)
    hesitation: int = Field(..., ge=0)
    retries: int = Field(..., ge=0)
    child_name: str = "Friend"

# ======================
# Helper: Generate Peco Message with Gemini
# ======================
def generate_peco_message(skill_level: str, accuracy: float, retries: int, child_name: str) -> str:
    prompt = f"""
You are Peco, a warm, friendly, and supportive learning companion for children.
Speak in short, simple, encouraging sentences (maximum 2 short sentences).
Do not use complicated words.

Child name: {child_name}
Skill level predicted: {skill_level}
Accuracy: {accuracy}%
Retries: {retries}

Generate a suitable message:
- If Beginner or low accuracy → be comforting and gentle
- If Intermediate → be encouraging
- If Advanced → be proud and happy
"""

    try:
        response = model_ai.generate_content(prompt)
        print("✅ Gemini Success:", response.text)
        return response.text.strip()
    except Exception as e:
        print("❌ Gemini Error:", str(e))
        # Fallback messages
        if skill_level == "Beginner":
            return f"It's okay, {child_name}. Let's try together!"
        elif skill_level == "Advanced":
            return f"Wow {child_name}! You did amazing!"
        else:
            return f"Great effort, {child_name}! Keep going!"
# ======================
# API Endpoint
# ======================
@app.post("/predict")
def predict(stats: GameStats):
    # 1. Predict skill level using your model
    data = pd.DataFrame(
        [[stats.accuracy, stats.reaction_time, stats.hesitation, stats.retries]],
        columns=FEATURE_NAMES
    )
    skill_level = skill_model.predict(data)[0]

    # 2. Generate natural Peco message using Gemini
    peco_message = generate_peco_message(
        skill_level=skill_level,
        accuracy=stats.accuracy,
        retries=stats.retries,
        child_name=stats.child_name
    )

    # 3. Decide Peco emotion
    if skill_level == "Beginner" or stats.accuracy < 45:
        peco_state = "comforting"
    elif skill_level == "Advanced":
        peco_state = "proud"
    else:
        peco_state = "encouraging"

    return {
        "skill_level": skill_level,
        "peco_state": peco_state,
        "peco_message": peco_message
    }

# ======================
# Run
# ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)