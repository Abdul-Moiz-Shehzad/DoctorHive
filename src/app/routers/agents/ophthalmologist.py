from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.params import Form
from typing import List, Dict, Any, Optional
import uvicorn
from src.app.models import Specialized_Agents_Diagnosis_Response
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from src.utils.utilities import _parse_initial_round_output, parse_specialist_response
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

@router.post("/api/agents/ophthalmologist/debate")
async def run_ophthalmological_debate(
    case_id: str = Form(..., description="Case ID"),
    history: List[Dict] = Form(..., description="History"),
    neurologist_response: str = Form(..., description="Neurologist response"),
    cardiologist_response: str = Form(..., description="Cardiologist response"),
    ophthalmologist_rag: str = Form(..., description="RAG"),
    model: str = Form(..., description="Backend model: 'gpt' or 'gemini'"),
):
    llm = get_llm(model)

    system_prompt = """
You are a senior Ophthalmologist participating in a multidisciplinary case discussion.
You previously made an ophthalmological diagnosis for the patient based on their symptoms, history, and ophthalmic reports.

Now you are asked to re-evaluate your diagnosis considering what other specialists have said.
Be clinical, logical, and realistic — imagine this as a genuine debate among consultants during a hospital case review.

Your objectives:
1. Critically analyze the reasoning and conclusions of the Neurologist and Cardiologist.
2. Compare their assessments with your own ophthalmological interpretation.
3. Decide whether:
   - to stand by your diagnosis,
   - to partially support or integrate their points, or
   - to revise your diagnosis entirely if their reasoning is stronger.
4. Adjust your *confidence level* accordingly — lower it if you accept others reasoning, raise it if your view is reinforced.
5. Base your reasoning on clinical evidence (reports, exam findings) and not on non-clinical factors.

If you still stand by your diagnosis or want to refine it further, you may ask **follow-up questions** to clarify clinically relevant missing details — but **do not** ask questions already covered by the GP (the GP follow-up history will be provided). Ask only if a specific ophthalmic detail is missing or unclear.

Output Format (must strictly follow):
confidence: <updated confidence level as a number or percentage>
diagnosis: <your final, possibly revised, ophthalmological diagnosis>
explanation: <your detailed reasoning — explicitly state if you support, disagree with, or modify your view based on others points>
follow_ups: <any additional follow-up questions you want to ask, comma-separated. If none, write 'None'>
"""

    messages = [SystemMessage(content=system_prompt)]

    for entry in history:
        role = entry.get("role", "user")
        content = entry.get("content", "")
        timestamp = entry.get("timestamp", "")
        formatted = f"[{timestamp}] {content}" if timestamp else content

        if role == "user":
            messages.append(HumanMessage(content=formatted))
        elif role == "assistant":
            messages.append(AIMessage(content=formatted))
        else:
            messages.append(HumanMessage(content=formatted))

    debate_prompt = f"""
Here is what other specialists concluded:
**Neurologist:** {neurologist_response}
**Cardiologist:** {cardiologist_response}

Your previous Ophthalmological diagnosis was informed by this context (RAG):
{ophthalmologist_rag}

Now, based on the full conversation history (which includes all user data, reports, and your past responses),
provide your final stance.

Remember:
- You can stand by your own diagnosis and defend it logically.
- You can agree with another specialist if their reasoning fits better clinically.
- Or you can modify your diagnosis if new insight makes more sense.
- Always reason like a real Ophthalmologist, not an AI summarizer.

Output strictly in this format:
confidence: <updated confidence level as a number or percentage>
diagnosis: <your final, possibly revised, ophthalmological diagnosis>
explanation: <your detailed reasoning — explicitly state if you support, disagree with, or modify your view based on others points>
follow_ups: <any additional follow-up questions you want to ask, comma-separated. If none, write 'None'>"""

    messages.append(HumanMessage(content=debate_prompt))

    response = llm.invoke(messages)
    text = response.content.strip()

    confidence, diagnosis, explanation, follow_ups = parse_specialist_response(text)

    return (
        Specialized_Agents_Diagnosis_Response(
            confidence=confidence,
            diagnosis=diagnosis,
            explanation=explanation,
        ),
        system_prompt + "\n" + debate_prompt,
        follow_ups,
    )

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("ophthalmologist:app", host="0.0.0.0", port=8001, reload=True)
