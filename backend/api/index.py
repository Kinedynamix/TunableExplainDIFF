import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class UserPreferences(BaseModel):
    topic: str
    difficulty: int

DIFFICULTY_PROMPTS = {
    1: "Explain using simple analogies suitable for a 10-year-old.",
    2: "Explain clearly for a layperson without using heavy jargon.",
    3: "Provide a standard, balanced explanation defining core technical terms.",
    4: "Explain at an undergraduate university level with formal concepts.",
    5: "Provide a rigorous graduate-level summary with technical constraints."
}

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.post("/api/explain")
def explain(prefs: UserPreferences):
    system_instruction = DIFFICULTY_PROMPTS.get(prefs.difficulty, DIFFICULTY_PROMPTS[3])

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",  # Replacement as the docs say
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Explain this concept: {prefs.topic}"}
            ],
            max_tokens=600
        )
        return {"explanation": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))