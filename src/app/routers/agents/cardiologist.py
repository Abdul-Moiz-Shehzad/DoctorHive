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
    title="Cardiologist Agent",
    description="Handles cardiovascular assessments, diagnoses, and treatment plans",
    version="2.0"
)
router = APIRouter(prefix="/agents/cardiologist")

def initial_round_cardiology_agent(user_message: str, backend: str = "gemini"):
    llm=get_llm(backend)
    system_prompt = """
You are a highly qualified Cardiologist.  
Your sole responsibility is to evaluate the patient's symptoms, history, and details strictly within the domain of cardiology.  
Do not provide information or diagnoses outside of cardiology, even if the query seems related to other medical fields.  
If the issue is clearly outside cardiology, respond with a cardiovascular perspective only (e.g., "Not within cardiology scope"). And in that case the confidence is 0.  

Your task is to provide:  
1. An approximate cardiovascular diagnosis.  
2. A confidence level as a numeric percentage between 0 and 100 (representing the likelihood of this diagnosis being correct).  
3. A clear explanation of the reasoning process that led to the diagnosis.  

Confidence should reflect both the clarity of the symptoms and the certainty of diagnosis without further tests.  
- High confidence (80-100): Classic, textbook presentation, very few alternatives.  
- Medium confidence (40-79): Common symptoms but multiple differential diagnoses possible.  
- Low confidence (1-39): Symptoms are vague, nonspecific, or overlap strongly with non-cardiological conditions.  
- Zero (0): Clearly outside cardiology.  

Always use this output format exactly (no extra words, no formatting changes):  
confidence: <number between 0 and 100>  
diagnosis: <diagnosis or None>  
explanation: <explanation>  

Example:  
User: I have a fever and a cough.  
confidence: 0  
diagnosis: None  
explanation: Not within cardiology scope.  

User: I have severe chest pain radiating to my left arm, with sweating and nausea.  
confidence: 90  
diagnosis: Acute myocardial infarction  
explanation: Sudden, severe chest pain radiating to the arm with autonomic symptoms is a classic presentation of myocardial infarction.  

User: My ankles are swollen, and I get short of breath when lying down at night.  
confidence: 75  
diagnosis: Congestive heart failure  
explanation: Peripheral edema and orthopnea are strongly suggestive of fluid overload in heart failure.  

User: Sometimes I feel a fluttering in my chest, especially after coffee or exercise.  
confidence: 55  
diagnosis: Atrial fibrillation or other arrhythmia  
explanation: Palpitations may indicate atrial fibrillation or other arrhythmias, but the description is nonspecific without ECG confirmation.  

User: I occasionally feel tired and dizzy, but I don't have chest pain or palpitations.  
confidence: 20  
diagnosis: None  
explanation: Fatigue and dizziness are nonspecific and may not indicate a cardiovascular disorder; other systemic causes are more likely.  
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



@router.post("/api/agents/cardiologist/assess", response_model=Specialized_Agents_Diagnosis_Response)
async def run_cardiological_diagnosis(
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
        initial_round_output = initial_round_cardiology_agent(input_data, model)
        return initial_round_output
    return {stage:"Not implemented yet"}

@router.post("/api/agents/cardiologist/debate")
async def run_cardiological_debate(
    case_id: str = Form(..., description="Case ID"),
    history: List[Dict] = Form(..., description="History"),
    neurologist_reponse: str = Form(..., description="Neurologist response"),
    ophthalmologist_response: str = Form(..., description="Ophthalmologist response"),
    cardiologist_rag: str = Form(..., description="RAG"),
    model: str = Form(..., description="Backend model: 'gpt' or 'gemini'"),
):
    llm = get_llm(model)

    system_prompt = """
You are a senior Cardiologist participating in a multidisciplinary case discussion.
You previously made a cardiological diagnosis for the patient based on their symptoms, history, and diagnostic reports.

Now, you are asked to re-evaluate your diagnosis considering what other specialists have said.
Be clinical, logical, and realistic — imagine this as a genuine debate between consultants in a hospital board meeting.

Your objectives:
1. Critically analyze the reasoning and conclusions of the Neurologist and Ophthalmologist.
2. Compare their assessments with your own cardiological interpretation.
3. Decide whether:
   - to stand by your diagnosis,
   - to partially support or integrate their points, or
   - to revise your diagnosis entirely if their reasoning is stronger.
4. Adjust your *confidence level* accordingly — lower it if you accept others reasoning, raise it if your stance is reinforced.
5. Focus on cardiovascular findings but acknowledge cross-specialty correlations (e.g., neurological or ophthalmic links).

If you still stand by your diagnosis or wish to refine it further, you may ask **follow-up questions** to clarify missing or uncertain details that could improve your final assessment.
However, **avoid asking questions already covered by the GP** (the GP's follow-up history will be provided). Only ask if a clinically relevant detail is missing or unclear.

Output Format (must strictly follow):
confidence: <updated confidence level as a number or percentage>
diagnosis: <your final, possibly revised, cardiological diagnosis>
explanation: <your detailed reasoning — mention whether you support, disagree, or modify based on others points, and justify your stance>
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
**Neurologist:** {neurologist_reponse}
**Ophthalmologist:** {ophthalmologist_response}

Your previous Cardiological diagnosis was informed by this context (RAG):
{cardiologist_rag}

Now, based on the full conversation history (which includes all user data, reports, and your past responses),
provide your final stance.

Remember:
- You can stand by your own diagnosis and defend it logically.
- You can agree with another specialist if their reasoning fits better clinically.
- Or you can modify your diagnosis if new insight makes more sense.
- Always reason like a real cardiologist, not an AI summarizer.

Output strictly in this format:
confidence: <updated confidence level as a number or percentage>
diagnosis: <your final, possibly revised, cardiological diagnosis>
explanation: <your detailed reasoning — mention whether you support, disagree, or modify based on others points, and justify your stance>
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
    uvicorn.run("cardiologist:app", host="0.0.0.0", port=8001, reload=True)
