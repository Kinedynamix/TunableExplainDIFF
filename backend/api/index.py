import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class UserPreferences(BaseModel):
    topic: str
    difficulty: int = Field(ge=1, le=5, default=3)
    model: str = "openai/gpt-oss-20b"
    max_tokens: int = Field(default=600, ge=300, le=1000)  # Min 300, Max 1000

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
            model=prefs.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Explain this concept: {prefs.topic}"}
            ],
            max_tokens=prefs.max_tokens  # Configurable per request
        )
        return {"explanation": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))