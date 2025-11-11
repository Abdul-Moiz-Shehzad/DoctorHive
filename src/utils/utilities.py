import re
from src.database import Base, engine, SessionLocal
import time
import logging
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.app.config import OPENAI_API_KEY, GOOGLE_API_KEY
logger = logging.getLogger(__name__)

def parse_specialist_response(text: str) -> dict:
    """
    Parses the model output text and extracts:
    - confidence (int or str)
    - diagnosis (str)
    - explanation (str)
    - follow_ups (list of str)
    Expected format:
        confidence: <number or percentage>
        diagnosis: <text>
        explanation: <text>
        follow_ups:
        1. <question one>
        2. <question two>
    """
    confidence = None
    diagnosis = None
    explanation = None
    follow_ups = []

    try:
        conf_match = re.search(r"confidence:\s*([^\n]+)", text, re.IGNORECASE)
        if conf_match:
            conf_text = conf_match.group(1).strip()
            try:
                confidence = int(re.sub(r"[^\d]", "", conf_text))  
            except ValueError:
                confidence = conf_text  

        diag_match = re.search(r"diagnosis:\s*([^\n]+)", text, re.IGNORECASE)
        if diag_match:
            diagnosis = diag_match.group(1).strip()

        expl_match = re.search(r"explanation:\s*(.+?)(?=\n\s*follow_ups:|\Z)", text, re.IGNORECASE | re.DOTALL)
        if expl_match:
            explanation = expl_match.group(1).strip()

        follow_section = re.search(r"follow_ups:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if follow_section:
            follow_text = follow_section.group(1).strip()
            if follow_text.lower() != "none":
                follow_ups = re.findall(r"^\s*\d+\.\s*(.+)", follow_text, re.MULTILINE)
                follow_ups = [q.strip() for q in follow_ups if q.strip()]
            else:
                follow_ups = []

    except Exception as e:
        explanation = f"Parsing error: {e}\nRaw output:\n{text}"

    return {
        "confidence": confidence,
        "diagnosis": diagnosis,
        "explanation": explanation,
        "follow_ups": follow_ups
    }

def parse_follow_ups(text: str):
    """
    Parses only the follow_ups from the model output text.
    """
    follow_ups = []
    try:
        lines = text.strip().split("\n")
        for line in lines:
            if line.lower().startswith("follow_ups:"):
                followup_text = line.split(":", 1)[1].strip()
                if followup_text.lower() != "none":
                    # Split on ? followed by optional whitespace
                    parts = re.split(r'\?\s*', followup_text)
                    follow_ups = [ (p.strip() + '?') for p in parts if p.strip() ]
                else:
                    follow_ups = []
                break  # Stop after finding the follow_ups line
    except Exception as e:
        logger.error(f"Parsing error: {e}\nRaw output:\n{text}")
        follow_ups = []  # Default to empty on error
    return follow_ups
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