import re
from src.database import Base, engine, SessionLocal
import time
import logging
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.app.config import OPENAI_API_KEY, GOOGLE_API_KEY, GOOGLE_API_KEY_second, GOOGLE_API_KEY_third, GOOGLE_API_KEY_fourth
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
    """Generate a short meaningful chat name from any user message."""

    # Skip system notes and find the actual patient complaint
    if '[System Note:' in message:
        complaint_match = re.search(r'Patient Complaint:\s*(.+)', message, re.IGNORECASE | re.DOTALL)
        if complaint_match:
            message = complaint_match.group(1).strip()

    # Lowercase message
    cleaned = message.lower().strip()

    # Remove common starting phrases only
    cleaned = re.sub(
        r'^(hi|hello|hey|doctor|dr|please|can you|could you|i have|i am|i feel|i need|my|the|patient says|complaint)\s+',
        '',
        cleaned,
        flags=re.IGNORECASE
    )

    # Replace punctuation with spaces
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)

    words = [w for w in cleaned.split() if len(w) > 1]

    # Common useless words that should not appear in title
    stop_words = {
        'i', 'me', 'my', 'we', 'our', 'you', 'your',
        'the', 'a', 'an', 'is', 'am', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'should', 'could', 'can',
        'which', 'that', 'this', 'these', 'those', 'also',
        'and', 'or', 'but', 'because', 'so', 'if', 'then',
        'for', 'to', 'of', 'in', 'on', 'at', 'by', 'with',
        'from', 'into', 'about', 'as', 'it', 'its',
        'please', 'help', 'doctor', 'dr', 'patient', 'says',
        'complaint', 'problem', 'issue', 'affecting', 'causing',
        'having', 'feeling', 'suffering'
    }

    # Important domain words from different categories
    important_keywords = {
        # Neuro
        'headache', 'migraine', 'dizziness', 'seizure', 'numbness',
        'weakness', 'memory', 'confusion', 'fainting',

        # Cardio
        'chest', 'heart', 'palpitation', 'palpitations', 'bp',
        'pressure', 'breathlessness', 'shortness', 'pulse',

        # Ophthalmology
        'eye', 'eyes', 'eyesight', 'vision', 'blurred', 'blurry',
        'redness', 'tearing', 'blindness',

        # General medical
        'fever', 'cough', 'pain', 'stomach', 'vomiting', 'nausea',
        'diarrhea', 'rash', 'swelling', 'bleeding', 'infection',
        'throat', 'fatigue', 'allergy',

        # Non-medical/general
        'appointment', 'report', 'medicine', 'prescription',
        'diet', 'sleep', 'stress', 'anxiety'
    }

    important_words = []
    normal_words = []

    for word in words:
        if word in stop_words:
            continue

        if word in important_keywords:
            important_words.append(word)
        else:
            normal_words.append(word)

    # Prefer important words, then add normal meaningful context
    name_words = []

    for word in important_words:
        if word not in name_words:
            name_words.append(word)

    for word in normal_words:
        if word not in name_words:
            name_words.append(word)

    # Final fallback
    if not name_words:
        name_words = ['Medical', 'Consultation']

    return ' '.join(word.capitalize() for word in name_words[:4])

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
        logger.info("Agent used Gemini with fallback rotation (Gemini 3 First)")
        
        # --- Gemini 2.5 Instances ---
        first_g25 = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=1,
            api_key=GOOGLE_API_KEY
        )
        second_g25 = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=1,
            api_key=GOOGLE_API_KEY_second
        )
        third_g25 = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=1,
            api_key=GOOGLE_API_KEY_third
        )
        fourth_g25 = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=1,
            api_key=GOOGLE_API_KEY_fourth
        )

        # --- Gemini 3 Instances ---
        first_g3 = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=1,
            api_key=GOOGLE_API_KEY
        )
        second_g3 = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=1,
            api_key=GOOGLE_API_KEY_second
        )
        third_g3 = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=1,
            api_key=GOOGLE_API_KEY_third
        )
        fourth_g3 = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=1,
            api_key=GOOGLE_API_KEY_fourth
        )

        # Chain: g3_1 -> g3_2 -> g3_3 -> g3_4 -> g25_1 -> g25_2 -> g25_3 -> g25_4
        llm_with_fallbacks = first_g3.with_fallbacks([
            second_g3,
            third_g3,
            fourth_g3,
            first_g25,
            second_g25,
            third_g25,
            fourth_g25
        ])

        return llm_with_fallbacks

    else:
        raise ValueError("Unsupported backend. Use 'gpt' or 'gemini'.")

def extract_content(content) -> str:
    """Robustly extracts text content from an LLM response content field."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            else:
                text_parts.append(str(part))
        return "".join(text_parts).strip()
    return str(content).strip()