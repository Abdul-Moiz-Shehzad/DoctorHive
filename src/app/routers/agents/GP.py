# GP.py
import os
import shutil
import sys
from fastapi import APIRouter, FastAPI, HTTPException, UploadFile
from fastapi.params import Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from src.app.models import UserInput, GPResponse
from src.app.config import UPLOAD_FOLDER, GOOGLE_API_KEY, OPENAI_API_KEY
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
import time
import re
# Remove the time.sleep from gemini of get_llm
# Deal with case state 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="General Physician Agent",
    description="Handles intake, follow-ups, and referral to specialists",
    version="1.0"
)
router = APIRouter(prefix="/agents/gp", tags=["GP Agent"])

def get_llm(backend: str = "gemini"):
    if backend == "gpt":
        logger.info("GP Agent used GPT")
        raise ValueError("I wont burn my money just yet")
        return ChatOpenAI(model="gpt-5-mini", temperature=1, api_key=OPENAI_API_KEY)
    elif backend == "gemini":
        logger.info("GP Agent used Gemini")
        time.sleep(30)
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1, api_key=GOOGLE_API_KEY)
    else:
        raise ValueError("Unsupported backend. Use 'gpt' or 'gemini'.")

def parse_gp_output(raw_output: str) -> dict:
    """
    Parse the LLM's raw plain text into GPResponse fields.
    """
    keyword = None
    response = None
    follow_ups = []
    specialists = None

    keyword_match = re.search(r"keyword:\s*(.+)", raw_output, re.IGNORECASE)
    if keyword_match:
        keyword = keyword_match.group(1).strip()

    response_match = re.search(r"response:\s*(.+)", raw_output, re.IGNORECASE | re.DOTALL)
    if response_match:
        resp_text = response_match.group(1).strip()
        resp_text = re.split(r"\n\s*1\.|\nspecialist:", resp_text, maxsplit=1)[0].strip()
        response = resp_text

    follow_ups = re.findall(r"^\s*\d+\.\s*(.+)", raw_output, re.MULTILINE)
    if not follow_ups:
        follow_ups = None

    specialist_match = re.search(r"specialist:\s*(.+)", raw_output, re.IGNORECASE)
    if specialist_match:
        spec_text = specialist_match.group(1).strip()
        if spec_text.lower() != "none":
            specialists = [s.strip() for s in spec_text.split(",")]
        else:
            specialists = None

    return {
        "keyword": keyword,
        "response": response,
        "follow_up_questions": follow_ups,
        "specialists_required": specialists
    }



def gp_agent(user_message: str, backend: str, case_state: Optional[Dict[str, Any]]) -> GPResponse:
    llm = get_llm(backend)

    system_prompt = """
You are a medical General Physician Doctor. 
STRICT RULES:
- Always communicate in plain text only. 
- Do NOT use tables, JSON, bullet symbols other than "1.", "2.", etc.
- Do NOT provide a diagnosis unless the case is very simple and no follow-up questions are needed.
- If the case is not that simple and requires Specialized Doctor out of [Neurologist, Cardiologist, Ophthalmologist] then return the specialized Doctor as well just keywords

Output formats:
1. If follow-up questions are required:
   keyword: follow-up-questions-gp
   response: <empathetic greeting + context>
   follow_up: 
   1. <question one>
   2. <question two>
   ...
   specialist: <sp1>, <sp3>

2. If the case is very simple and no follow-up questions are needed:
   keyword: direct
   response: <empathetic short reply>
   follow_up: None
   specialist: None

Do not change these formats under any circumstance.
"""

    prompt = f"{system_prompt}\n\nPatient says: {user_message}"
    try:
        raw_response = llm.invoke(prompt).content.strip()
        logger.info(f"LLM Raw response: {raw_response}")
    except Exception as e:
        logger.error(f"LLM error: {e}")
        raise HTTPException(status_code=500, detail="LLM backend error")

    parsed_output = parse_gp_output(raw_response)
    logger.info(f"Parsed output: {parsed_output}")

    return GPResponse(
        keyword=parsed_output.get("keyword"),
        response=parsed_output.get("response"),
        follow_up_questions=parsed_output.get("follow_up_questions"),
        specialists_required=parsed_output.get("specialists_required")
    )


@app.post("/api/agents/gp/assess", response_model=GPResponse)
async def GP_assess_case(
    message: str = Form(...,description="Patient message input"),
    files_content: Optional[str] = Form("", description="Optional extra files content"),
    model:str = Form(..., description="Backend model: 'gpt' or 'gemini'"),
):
    """
    Assess a patient case. Accepts:
    - message (required)
    - file content parsed (optional, e.g. PDF or image reports)
    - model (required gpt or gemini)

    Output Format:
    - keyword: <keyword> [Direct, follow-up-questions-gp]
    - response: <response>
    - follow_up_questions: <follow_up_questions> [Questions or None]
    - specialists_required: <specialists_required> [Specialist or None]
    """
    # Example:
    #         {
    #     "keyword": "follow-up-questions-gp",
    #     "response": "I understand that you're experiencing episodes where the central part of your vision blacks out, lasting for a few minutes. This sounds concerning, and I want to gather more information to understand what might be happening.\nfollow_up:",
    #     "follow_up_questions": [
    #         "How frequently do these episodes occur?",
    #         "Do you experience any other symptoms before, during, or after these episodes, such as headaches, dizziness, nausea, or visual disturbances like flashing lights or shimmering?",
    #         "Do you have any family history of eye problems, migraines, or neurological conditions?",
    #         "Have you had your blood pressure checked recently?",
    #         "Do you have diabetes or any other medical conditions?",
    #         "What medications are you currently taking, including over-the-counter medications and supplements?"
    #     ],
    #     "specialists_required": [
    #         "Neurologist",
    #         "Ophthalmologist"
    #     ]
    # }
    try:
        if model not in ["gpt", "gemini"]:
            logger.error(f"Invalid model specified: {model}")
            raise HTTPException(status_code=400, detail="Invalid model specified.")

        input_data = message + files_content
        response = gp_agent(input_data, model, case_state={})
        return response

    except Exception as e:
        logger.error(f"Error in GP assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router)

@app.on_event("shutdown")
def cleanup_uploads():
    """Delete upload folder when app exits."""
    try:
        if "reload" in sys.argv:
            logger.info("Skipping cleanup during autoreload.")
            return
        elif os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            logger.info(f"Upload folder {UPLOAD_FOLDER} deleted on shutdown.")
    except Exception as e:
        logger.error(f"Failed to clean upload folder: {e}")

if __name__ == "__main__":
    uvicorn.run("GP:app", host="0.0.0.0", port=8001, reload=True)
