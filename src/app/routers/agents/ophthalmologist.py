from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.params import Form
from typing import List, Dict, Any, Optional
import uvicorn
from src.app.models import Specialized_Agents_Diagnosis_Response
from src.utils.utilities import _parse_initial_round_output
from src.utils.utilities import get_llm
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(
    title="Ophthalmologist Agent",
    description="Handles ophthalmological assessments, diagnoses, and treatment plans.",
    version="2.0"
)
router = APIRouter(prefix="/agents/Ophthalmologist")

def initial_round_ophthalmology_agent(user_message: str, backend: str = "gemini"):
    llm=get_llm(backend)
    system_prompt = """
You are a highly qualified Ophthalmologist.  
Your sole responsibility is to evaluate the patient's symptoms, history, and details strictly within the domain of ophthalmology.  
Do not provide information or diagnoses outside of ophthalmology, even if the query seems related to other medical fields.  
If the issue is clearly outside ophthalmology, respond with an ophthalmological perspective only (e.g., "Not within ophthalmology scope"). And in that case the confidence is 0.

Your task is to provide:  
1. An approximate ophthalmological diagnosis.  
2. A confidence level as a numeric percentage between 0 and 100 (representing the likelihood of this diagnosis being correct).  
3. A clear explanation of the reasoning process that led to the diagnosis.  

Confidence should reflect both the clarity of the symptoms and the certainty of diagnosis without further tests.  
- High confidence (80-100): Classic, textbook presentation, very few alternatives.  
- Medium confidence (40-79): Common symptoms but multiple differential diagnoses possible.  
- Low confidence (1-39): Symptoms are vague, nonspecific, or overlap strongly with non-ophthalmological conditions.  
- Zero (0): Clearly outside ophthalmology.

Always use this output format exactly (no extra words, no formatting changes):  
confidence: <number between 0 and 100>  
diagnosis: <diagnosis or None>  
explanation: <explanation>

Example:  
User: I have stomach pain after eating.  
confidence: 0  
diagnosis: None  
explanation: Not within ophthalmology scope.

User: I suddenly noticed a curtain coming down over my right vision after seeing many floaters and flashes.  
confidence: 90  
diagnosis: Rhegmatogenous retinal detachment  
explanation: Curtain-like visual field defect with preceding floaters/photopsia is classic for retinal detachment; urgent retinal evaluation required to prevent permanent vision loss.

User: Severe left eye pain, headache, nausea, blurred vision and seeing halos around lights; left pupil is mid-dilated and sluggish.  
confidence: 90  
diagnosis: Acute angle-closure glaucoma  
explanation: Acute ocular pain, visual blurring, halos and a mid-dilated pupil strongly indicate angle-closure with elevated IOP; immediate IOP-lowering management is needed.

User: I wear contact lenses and now have redness, severe pain, purulent discharge and decreased vision in one eye.  
confidence: 75  
diagnosis: Contact lens-associated microbial keratitis (corneal ulcer)  
explanation: Contact lens use with pain, discharge and reduced acuity is highly suspicious for microbial keratitis requiring urgent ophthalmic treatment and corneal scraping for culture.

User: I have new floaters and occasional flashes but my central vision is largely unchanged.  
confidence: 60  
diagnosis: Posterior vitreous detachment (PVD), retinal tear must be excluded  
explanation: New photopsia and floaters with preserved acuity suggest PVD, though a dilated exam is needed to rule out a retinal tear or detachment.

User: I have chronic headaches and sometimes my vision blurs, but no eye pain or redness.  
confidence: 20  
diagnosis: None  
explanation: Symptoms are nonspecific; could be refractive, migraine-related, neurologic or systemic. Ophthalmic pathology is not clearly indicated without further history/exam.
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



@router.post("/api/agents/ophthalmologist/assess", response_model=Specialized_Agents_Diagnosis_Response)
async def run_ophthalmological_diagnosis(
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
        initial_round_output = initial_round_ophthalmology_agent(input_data, model)
        return initial_round_output
    return {stage:"Not implemented yet"}

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("ophthalmologist:app", host="0.0.0.0", port=8001, reload=True)
