import re
from src.database import Base, engine, SessionLocal
import time
import logging
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.app.config import OPENAI_API_KEY, GOOGLE_API_KEY, GOOGLE_API_KEY_second, GOOGLE_API_KEY_third
logger = logging.getLogger(__name__)

def clean_question(q: str) -> str:
    """Removes leading non-alphanumeric characters from a question string."""
    if not q:
        return q
    # Remove leading characters that aren't letters or numbers (like commas, dots, spaces, etc.)
    # We use a more explicit regex to catch common artifacts
    cleaned = re.sub(r'^[,\s.\-!]+', '', q.strip())
    # Capitalize the first letter if it exists
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned

def generate_chat_name(message: str) -> str:
    """Generate a short chat name (3-4 words max) based on the user's message."""
    # Skip system notes and find the actual patient complaint
    if '[System Note:' in message:
        # Find the patient complaint section
        complaint_match = re.search(r'Patient Complaint:\s*(.+)', message, re.IGNORECASE | re.DOTALL)
        if complaint_match:
            message = complaint_match.group(1).strip()
    
    # Remove common medical prefixes and clean the message
    cleaned = re.sub(r'^(hi|hello|doctor|i have|i am|i feel|my|the|please help|help with|patient says|complaint)\s*', '', message.lower(), flags=re.IGNORECASE)
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)  # Replace punctuation with spaces
    words = [w for w in cleaned.split() if len(w) > 1]  # Filter out single letters
    
    # Look for medical symptoms/conditions (prioritize these)
    medical_keywords = ['pain', 'headache', 'chest', 'stomach', 'back', 'joint', 'muscle', 'fever', 'cough', 'nausea', 'dizziness', 'fatigue', 'rash', 'swelling', 'bleeding', 'infection', 'inflammation']
    
    medical_words = []
    other_words = []
    
    for word in words:
        if any(keyword in word for keyword in medical_keywords):
            medical_words.append(word)
        else:
            other_words.append(word)
    
    # Prefer medical terms, but include some context
    if medical_words:
        if len(medical_words) >= 2:
            name_words = medical_words[:3]  # Take up to 3 medical words
        else:
            # If only one medical word, take it plus one context word if available
            name_words = medical_words
            if other_words and len(other_words) > 0:
                # Find a relevant context word (avoid pronouns, articles)
                context_words = [w for w in other_words if w not in ['i', 'my', 'the', 'a', 'an', 'doctor', 'hello', 'hi']]
                if context_words:
                    name_words.append(context_words[0])
    else:
        # Fallback to first meaningful words
        name_words = other_words[:3]
    
    if not name_words:
        name_words = words[:3] if words else ['Medical', 'Consultation']
    
    # Capitalize first letter of each word
    name = ' '.join(word.capitalize() for word in name_words[:4])
    
    return name

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
                follow_ups = [clean_question(q) for q in follow_ups if q.strip()]
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
                    follow_ups = [ (clean_question(p) + '?') for p in parts if p.strip() ]
                else:
                    follow_ups = []
                break  # Stop after finding the follow_ups line
    except Exception as e:
        logger.error(f"Parsing error: {e}\nRaw output:\n{text}")
        follow_ups = []  # Default to empty on error
    return follow_ups

def _parse_gp_output(raw_output: str) -> dict:
    """
    Parse the LLM's raw plain text into GPResponse fields.
    Robustly handles missing tags by defaulting to 'direct' keyword and using raw output as response.
    """
    keyword = None
    response = None
    follow_ups = []
    specialists = None

    # 1. Try to find keyword
    keyword_match = re.search(r"keyword:\s*(.+)", raw_output, re.IGNORECASE)
    if keyword_match:
        keyword = keyword_match.group(1).strip()

    # 2. Try to find response
    response_match = re.search(r"response:\s*(.+)", raw_output, re.IGNORECASE | re.DOTALL)
    if response_match:
        resp_text = response_match.group(1).strip()
        # Clean up if it captures until next sections
        resp_text = re.split(r"\n\s*1\.|\nspecialist:", resp_text, maxsplit=1)[0].strip()
        response = resp_text
        if response:
            response = response.replace("\nfollow_up:", "")

    # 3. Robust fallbacks for missing tags (common in off-topic or refusal responses)
    if not keyword:
        keyword = "direct"
    
    if not response:
        # If 'response:' tag is missing, use the raw output as the response
        # If 'keyword:' was found separately, try to exclude that line
        if keyword_match:
            response = re.sub(r"keyword:\s*.+", "", raw_output, count=1, flags=re.IGNORECASE).strip()
        else:
            response = raw_output.strip()
        
        # Final fallback to empty string if raw_output was somehow empty
        if not response:
            response = "No response generated."

    # 4. Parse follow-ups
    follow_ups = [clean_question(q) for q in re.findall(r"^\s*\d+\.\s*(.+)", raw_output, re.MULTILINE)]
    if not follow_ups:
        follow_ups = None

    # 5. Parse specialists
    specialist_match = re.search(r"specialist:\s*(.+)", raw_output, re.IGNORECASE)
    if specialist_match:
        spec_text = specialist_match.group(1).strip()
        if spec_text.lower() != "none" and spec_text:
            specialists = [s.strip() for s in spec_text.split(",")]
        else:
            specialists = None

    return {
        "keyword": keyword,
        "response": response,
        "follow_up_questions": follow_ups,
        "specialists_required": specialists
    }
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
        return ChatOpenAI(model="gpt-4o", temperature=1, api_key=OPENAI_API_KEY)
    
    elif backend == "gemini":
        logger.info("Agent used Gemini with fallback rotation")
        
        # Define your different instances
        primary = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1, api_key=GOOGLE_API_KEY)
        secondary = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1, api_key=GOOGLE_API_KEY_second)
        third = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1, api_key=GOOGLE_API_KEY_third)
        
        # Create a chain that automatically falls back on ResourceExhausted errors
        llm_with_fallbacks = primary.with_fallbacks([secondary, third])
        
        return llm_with_fallbacks

    else:
        raise ValueError("Unsupported backend. Use 'gpt' or 'gemini'.")