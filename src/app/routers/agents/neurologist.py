from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.params import Form
from typing import List, Dict, Any, Optional, Tuple
import uvicorn
from src.app.models import Specialized_Agents_Diagnosis_Response
from src.utils.utilities import _parse_initial_round_output
from src.utils.utilities import get_llm
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(
    title="Neurologist Agent",
    description="Handles neurological assessments, diagnoses, and treatment plans",
    version="2.0"
)
router = APIRouter(prefix="/agents/neurologist")

def initial_round_neurology_agent(user_message: str, backend: str = "gemini"):
    llm=get_llm(backend)
    system_prompt = """
You are a highly qualified Neurologist.  
Your sole responsibility is to evaluate the patient's symptoms, history, and details strictly within the domain of neurology.  
Do not provide information or diagnoses outside of neurology, even if the query seems related to other medical fields.  
If the issue is clearly outside neurology, respond with a neurological perspective only (e.g., "Not within neurology scope"). And in that case confidence is 0. 

Your task is to provide:  
1. An approximate neurological diagnosis.  
2. A confidence level as a numeric percentage between 0 and 100 (representing the likelihood of this diagnosis being correct).  
3. A clear explanation of the reasoning process that led to the diagnosis.  

Confidence should reflect both the clarity of the symptoms and the certainty of diagnosis without further tests.
- High confidence (80-100): Classic, textbook presentation, very few alternatives.
- Medium confidence (40-79): Common symptoms but multiple differential diagnoses possible.
- Low confidence (1-39): Symptoms are vague, nonspecific, or overlap strongly with non-neurological conditions.
- Zero (0): Clearly outside neurology.

Always use this output format exactly (no extra words, no formatting changes):  
confidence: <number between 0 and 100>
diagnosis: <diagnosis or None>  
explanation: <explanation>

Example:
User: I have stomach pain after eating.  
confidence: 0  
diagnosis: None  
explanation: Not within neurology scope.

User: Suddenly, my speech became slurred and my right arm feels weak.  
confidence: 90  
diagnosis: Acute ischemic stroke  
explanation: Sudden onset of unilateral weakness and speech disturbance is highly suggestive of stroke.  

User: I get headaches on one side of my head with nausea and sensitivity to light.  
confidence: 70  
diagnosis: Migraine  
explanation: Unilateral headache with nausea and photophobia is typical of migraine, though other causes should be ruled out.  

User: I sometimes lose awareness for a few seconds, and people say I stare blankly.  
confidence: 60  
diagnosis: Absence seizures (epilepsy)  
explanation: Brief episodes of impaired awareness and staring are consistent with absence seizures, but EEG is required for confirmation.  

User: I often feel dizzy and lightheaded when I stand up quickly.  
confidence: 25  
diagnosis: Possible orthostatic dizziness (not clearly neurological)  
explanation: Dizziness with posture changes is more often cardiovascular or vestibular, not central neurological. Neurological cause cannot be excluded but is unlikely.  
"""
    prompt=f"{system_prompt}\n\nPatient says: {user_message}"
    try:
        raw_response = llm.invoke(prompt).content.strip()
        logger.info(f"LLM Raw response: {raw_response}")
    except Exception as e:
        logger.error(f"LLM error: {e}")
        raise HTTPException(status_code=500, detail="LLM backend error")
    parsed_output = _parse_initial_round_output(raw_response)
    return parsed_output



@router.post("/api/agents/neurologist/assess", response_model=Specialized_Agents_Diagnosis_Response)
async def run_neurological_diagnosis(
    message: str = Form(...,description="Patient message input"),
    followup_history: str = Form(..., description="follow-up history"),
    files_content: Optional[str] = Form("", description="Optional extra files content"),
    model:str = Form(..., description="Backend model: 'gpt' or 'gemini'"),
    stage: str=Form(...,description="Case stage"),
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
    # Message has user's query AND follow-up questions asked from the GP
    # files_content has the content of files parsed into it
    input_data = (
    "Patient message:\n"
    f"{message}\n\n"
    "Follow-up history (questions and answers from previous agents):\n"
    f"{followup_history}\n\n"
    "Attached medical reports (parsed content):\n"
    f"{files_content}"
)
    if stage=="initial":
        initial_round_output = initial_round_neurology_agent(input_data, model)
        return initial_round_output
    return {stage:"Not implemented yet"}



@router.post("/api/agents/neurologist/debate", response_model=Specialized_Agents_Diagnosis_Response)
async def run_neurological_debate(
    case_id: str = Form(...,description="Case ID"),
    history: List[Dict] = Form(...,description="History"),
    cardiologist_response: str = Form(...,description="Cardiologist response"),
    ophthalmologist_response: str = Form(...,description="Ophthalmologist response"),
    neurologist_rag: str = Form(...,description="RAG"),
    model: str = Form(...,description="Backend model: 'gpt' or 'gemini'"),
):
    llm=get_llm(model)
    
    return None

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("neurologist:app", host="0.0.0.0", port=8001, reload=True)
