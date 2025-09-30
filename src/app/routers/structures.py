import logging
from app.routers.agents.cardiologist import run_cardiological_diagnosis
from app.routers.agents.ophthalmologist import run_ophthalmological_diagnosis
from fastapi import APIRouter, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from src.app.models import CardiologistHistory, Case, NeurologistHistory, OphthalmologistHistory
from src.app.routers.agents.neurologist import run_neurological_diagnosis
from datetime import datetime
from src.utils.utilities import get_db

router = APIRouter(prefix="/structures")
logger=logging.getLogger(__name__)

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
    followup_history = (
    db.query(Case.answered_followups).filter(Case.case_id == case_id).scalar())
    files_content = db.query(Case.answered_followups).filter(Case.case_id == case_id).scalar()
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

@router.post("/first_debate_round")
async def first_debate_round(
    case_id: str = Form(...),
    message: str = Form(...),
    files_content: str = Form("None"),
    model: str = Form(...),
    db: Session = Depends(get_db),
):
    check_case = db.query(Case).filter(Case.case_id == case_id).first()
    if not check_case:
        raise HTTPException(status_code=404, detail="Case not found")
    followup_history = (db.query(Case.answered_followups).filter(Case.case_id == case_id).scalar())