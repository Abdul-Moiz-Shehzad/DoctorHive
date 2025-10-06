import re
from src.database import Base, engine, SessionLocal
import time
import logging
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.app.config import OPENAI_API_KEY, GOOGLE_API_KEY
logger = logging.getLogger(__name__)
def _parse_initial_round_output(raw_output: str) -> dict:
    """Parse the LLM's raw plain text into SpecialistResponse Diagnosis fields."""
    confidence = None
    diagnosis = None
    explanation = None
    
    confidence_match = re.search(r"confidence:\s*(.+)", raw_output, re.IGNORECASE)
    if confidence_match:
        confidence = int(confidence_match.group(1).strip())

    diagnosis_match = re.search(r"diagnosis:\s*(.+)", raw_output, re.IGNORECASE)
    if diagnosis_match:
        diagnosis = diagnosis_match.group(1).strip()

    explanation_match = re.search(r"explanation:\s*(.+)", raw_output, re.IGNORECASE)
    if explanation_match:
        explanation = explanation_match.group(1).strip()

    return {
        "confidence": confidence,
        "diagnosis": diagnosis,
        "explanation": explanation
    }

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_llm(backend: str = "gemini"):
    if backend == "gpt":
        logger.info("Agent used GPT")
        raise ValueError("I wont burn my money just yet")
        return ChatOpenAI(model="gpt-5-mini", temperature=1, api_key=OPENAI_API_KEY)
    elif backend == "gemini":
        logger.info("Agent used Gemini")
        time.sleep(30)
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1, api_key=GOOGLE_API_KEY)
    else:
        raise ValueError("Unsupported backend. Use 'gpt' or 'gemini'.")