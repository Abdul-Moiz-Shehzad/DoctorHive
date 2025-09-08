# neurologist.py
import os
import shutil
import sys
from fastapi import APIRouter, FastAPI, HTTPException, UploadFile
from fastapi.params import Form
from fastapi.responses import RedirectResponse
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
    title="Neurologist Agent",
    description="Handles neurological assessments, diagnoses, and treatment plans",
    version="1.0"
)
router = APIRouter(prefix="/agents/neurologist")
def get_llm(backend: str = "gemini"):
    if backend == "gpt":
        logger.info("Neurologist Agent used GPT")
        raise ValueError("I wont burn my money just yet")
        return ChatOpenAI(model="gpt-5-mini", temperature=1, api_key=OPENAI_API_KEY)
    elif backend == "gemini":
        logger.info("Neurologist Agent used Gemini")
        time.sleep(30)
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1, api_key=GOOGLE_API_KEY)
    else:
        raise ValueError("Unsupported backend. Use 'gpt' or 'gemini'.")


@router.post("/api/agents/neurologist/assess", response_model=GPResponse)
async def assess_neurological_condition(
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