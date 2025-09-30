from fastapi import APIRouter, FastAPI, HTTPException, Depends
from fastapi.params import Form
from typing import List, Dict, Any, Optional
import uvicorn
from src.utils.utilities import get_db, get_llm
from sqlalchemy.orm import Session
from src.app.models import Case, GPResponse
import logging
import re
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="General Physician Agent",
    description="Handles intake, follow-ups, and referral to specialists",
    version="2.0"
)
router = APIRouter(prefix="/agents/gp")

def _parse_gp_output(raw_output: str) -> dict:
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
        response=response.replace("\nfollow_up:","")

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
- You are not allowed to answer questions out of medical domain.
- If user message is not related to medical regardless of the report attached or not then in that case you can not answer to that politely say it and no follow ups and specialist.
- Always communicate in plain text only. 
- Do NOT use tables, JSON, bullet symbols other than "1.", "2.", etc.
- Do NOT provide a diagnosis unless the case is very simple and no follow_up questions are needed.
- If the case is not that simple and requires Specialized Doctor out of [Neurologist, Cardiologist, Ophthalmologist] then return the specialized Doctor as well just keywords
- Output should include the keyword, response, follow_up, specialist in the exact format given in examples below.

Allowed keywords: direct, follow_up
Available Specialized Doctors: Neurologist, Cardiologist, Ophthalmologist

Output formats:
1. If follow_up questions are required:
   keyword: follow_up
   response: <empathetic greeting + details>
   follow_up: 
   1. <question one>
   2. <question two>
   ...
   specialist: <sp1>, <sp3>

2. If the case is very simple and no follow_up questions are needed:
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

    parsed_output = _parse_gp_output(raw_response)

    return GPResponse(
        keyword=parsed_output.get("keyword"),
        response=parsed_output.get("response"),
        follow_up_questions=parsed_output.get("follow_up_questions"),
        specialists_required=parsed_output.get("specialists_required")
    )


@router.post("/api/agents/gp/assess", response_model=GPResponse)
async def GP_assess_case(
    message: str = Form(...,description="Patient message input"),
    case_id: str = Form(...,description="Case ID"),
    model:str = Form(..., description="Backend model: 'gpt' or 'gemini'"),
    db: Session = Depends(get_db),
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
    case = db.query(Case).filter(Case.case_id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case ID not found")
    files_content = case.files_content
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

if __name__ == "__main__":
    uvicorn.run("GP:app", host="0.0.0.0", port=8001, reload=True)
