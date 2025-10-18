import json
import logging
from app.routers.agents.cardiologist import run_cardiological_debate, run_cardiological_diagnosis
from app.routers.agents.ophthalmologist import run_ophthalmological_debate, run_ophthalmological_diagnosis
from fastapi import APIRouter, Body, FastAPI, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from src.app.models import CardiologistHistory, Case, NeurologistHistory, OphthalmologistHistory
from src.app.routers.agents.neurologist import run_neurological_debate, run_neurological_diagnosis
from datetime import datetime
from src.utils.utilities import get_db

router = APIRouter(prefix="/structures")
logger=logging.getLogger(__name__)
app = FastAPI(
    title="Structure for Specialists",
    description="Handles all the flow of specialists",
    version="2.0"
)
@router.post("/initial_round")
async def initial_round(
    case_id: str = Form(...),
    message: str = Form(...),
    model: str = Form(...),
    db: Session = Depends(get_db),
):
    check_case = db.query(Case).filter(Case.case_id == case_id).first()
    if not check_case:
        raise HTTPException(status_code=404, detail="Case not found")
    followup_history = (db.query(Case.answered_followups).filter(Case.case_id == case_id).scalar())
    files_content = db.query(Case.files_content).filter(Case.case_id == case_id).scalar()
    files_content = str(files_content) if files_content is not None else ""
    input_data = (
        f"Patient message:\n{message}\n\n"
        f"Attached reports:\n{files_content}"
        f"\n\nFollow-up history:\n{followup_history}"
    )
    logger.info(f"Calling neurologist for case {case_id}")
    stage="initial"
    neuro_response = await run_neurological_diagnosis(message,followup_history,files_content, model,stage)
    cardio_response = await run_cardiological_diagnosis(message,followup_history,files_content, model,stage)
    ophthal_response = await run_ophthalmological_diagnosis(message,followup_history,files_content, model,stage)

    neuro_entry = NeurologistHistory(
        case_id=case_id,
        user_input=input_data,
        agent_response=neuro_response,
        timestamp=datetime.utcnow(),
    )
    db.add(neuro_entry)
    db.commit()
    db.refresh(neuro_entry)

    cardio_entry = CardiologistHistory(
        case_id=case_id,
        user_input=input_data,
        agent_response=cardio_response,
        timestamp=datetime.utcnow(),
    )
    db.add(cardio_entry)
    db.commit()
    db.refresh(cardio_entry)

    ophthal_entry = OphthalmologistHistory(
        case_id=case_id,
        user_input=input_data,
        agent_response=ophthal_response,
        timestamp=datetime.utcnow(),
    )
    db.add(ophthal_entry)
    db.commit()
    db.refresh(ophthal_entry)
    return {
        "case_id": case_id,
        "responses":{"neurologist":neuro_entry.agent_response,"cardiologist":cardio_entry.agent_response,"ophthalmologist":ophthal_entry.agent_response}
    }

def _build_history(case_id: str, agent: str, db):
    """
    Builds chat history for a given agent and returns it 
    in a LangChain-compatible format with timestamps.
    """

    agent_models = {
        "neurologist": NeurologistHistory,
        "cardiologist": CardiologistHistory,
        "ophthalmologist": OphthalmologistHistory,
    }

    model = agent_models.get(agent)
    if not model:
        raise ValueError(f"Unsupported agent type: {agent}")

    records = (
        db.query(model.user_input, model.agent_response, model.timestamp)
        .filter(model.case_id == case_id)
        .order_by(model.timestamp.asc())
        .all()
    )

    history = []
    for record in records:
        user_input, agent_response, timestamp = record

        history.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else timestamp,
        })

        history.append({
            "role": "assistant",
            "content": agent_response,
            "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else timestamp,
        })

    return history

def _add_follow_ups_for_db(neurologist_response_after_debate,cardiologist_response_after_debate,ophthalmologist_response_after_debate,neurologist_follow_ups,cardiologist_follow_ups,ophthalmologist_follow_ups):
    """Add the follow up questions asked by specialists to the database"""
    neurologist_response_after_debate=neurologist_response_after_debate.dict()
    cardiologist_response_after_debate=cardiologist_response_after_debate.dict()
    ophthalmologist_response_after_debate=ophthalmologist_response_after_debate.dict()
    neurologist_response_after_debate["follow_ups"]=neurologist_follow_ups
    cardiologist_response_after_debate["follow_ups"]=cardiologist_follow_ups
    ophthalmologist_response_after_debate["follow_ups"]=ophthalmologist_follow_ups
    return neurologist_response_after_debate,cardiologist_response_after_debate,ophthalmologist_response_after_debate

@router.post("/first_debate_round")
async def first_debate_round(
    case_id: str = Form(...),
    model: str = Form(...),
    initial_responses: str | dict = Form(...),
    db: Session = Depends(get_db),
):
    """Specialist agents will debate on the diagnosis of each."""
    if isinstance(initial_responses, str):
        try:
            initial_responses = json.loads(initial_responses)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in initial_responses")
    elif not isinstance(initial_responses, dict):
        raise HTTPException(status_code=400, detail="initial_responses must be a dict or JSON string")
    check_case = db.query(Case).filter(Case.case_id == case_id).first()
    if not check_case:
        raise HTTPException(status_code=404, detail="Case not found")
    # follow up history not needed because already stored in database for each agent's chat history.
    #followup_history = (db.query(Case.answered_followups).filter(Case.case_id == case_id).scalar())
    neurologist_response = initial_responses["neurologist"]
    cardiologist_response = initial_responses["cardiologist"]
    ophthalmologist_response = initial_responses["ophthalmologist"]
    neurologist_rag = None # TODO: RAG for neurologist
    cardiologist_rag = None # TODO: RAG for cardiologist
    ophthalmologist_rag = None # TODO: RAG for ophthalmologist

    # Build chat history for each agent
    neurologist_history = _build_history(case_id,"neurologist",db)
    cardiologist_history = _build_history(case_id,"cardiologist",db)
    ophthalmologist_history = _build_history(case_id,"ophthalmologist",db)

    # Starting debate
    neurologist_response_after_debate, neurologist_debate_prompt, neurologist_follow_ups = await run_neurological_debate(case_id,neurologist_history,cardiologist_response,ophthalmologist_response,neurologist_rag,model)
    cardiologist_response_after_debate, cardiologist_debate_prompt, cardiologist_follow_ups = await run_cardiological_debate(case_id,cardiologist_history,neurologist_response,ophthalmologist_response,cardiologist_rag,model)
    ophthalmologist_response_after_debate, ophthalmologist_debate_prompt, ophthalmologist_follow_ups = await run_ophthalmological_debate(case_id,ophthalmologist_history,neurologist_response,cardiologist_response,ophthalmologist_rag,model)
    
    neurologist_response_after_debate,cardiologist_response_after_debate,ophthalmologist_response_after_debate = _add_follow_ups_for_db(neurologist_response_after_debate,cardiologist_response_after_debate,ophthalmologist_response_after_debate,neurologist_follow_ups,cardiologist_follow_ups,ophthalmologist_follow_ups)
    neuro_entry = NeurologistHistory(
        case_id=case_id,
        user_input=neurologist_debate_prompt,
        agent_response=neurologist_response_after_debate,
        timestamp=datetime.utcnow(),
    )
    db.add(neuro_entry)
    db.commit()
    db.refresh(neuro_entry)

    cardio_entry = CardiologistHistory(
        case_id=case_id,
        user_input=cardiologist_debate_prompt,
        agent_response=cardiologist_response_after_debate,
        timestamp=datetime.utcnow(),
    )
    db.add(cardio_entry)
    db.commit()
    db.refresh(cardio_entry)

    ophthal_entry = OphthalmologistHistory(
        case_id=case_id,
        user_input=ophthalmologist_debate_prompt,
        agent_response=ophthalmologist_response_after_debate,
        timestamp=datetime.utcnow(),
    )
    db.add(ophthal_entry)
    db.commit()
    db.refresh(ophthal_entry)
    
    return {
        case_id:case_id,
        "responses":{"neurologist":neurologist_response_after_debate,"cardiologist":cardiologist_response_after_debate,"ophthalmologist":ophthalmologist_response_after_debate},
        "follow_ups":{"neurologist":neurologist_follow_ups,"cardiologist":cardiologist_follow_ups,"ophthalmologist":ophthalmologist_follow_ups}
    } 



app.include_router(router)