from datetime import datetime
import logging
import os
import shutil
import sys
import uuid
from src.app.routers.structures import debate_round, initial_round, specialists_improved_diagnosis, determine_consensus_winner
from fastapi import APIRouter, FastAPI, HTTPException, Form, UploadFile, Depends
from typing import Optional, List
from sqlalchemy import text
from sqlalchemy.orm import Session
 
from src.app.models import (
    InitialOrchestratorResponse,
    FollowUpResponse,
    FollowUpResponseSpecialists,
    Case,
    NeurologistHistory,
    CardiologistHistory,
    OphthalmologistHistory,
)
from src.app.routers.agents.GP import GP_assess_case
from src.utils.parse.parse_file import parse_endpoint
from src.app.config import UPLOAD_FOLDER
from src.database import Base, engine, SessionLocal
from src.utils.utilities import get_db, invoke_with_retry

app = FastAPI(
    title="Orchestrator",
    description="Handles all the flow",
    version="2.0"
)
router = APIRouter(prefix="/orchestrator")
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)
@app.on_event("startup")
async def startup_event():
    """Check if DB connection works before fully starting the app."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection successful.")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        import sys
        sys.exit(1)

@router.post("/process", response_model=InitialOrchestratorResponse)
async def process_patient_message_and_files(
    message: str = Form(..., description="Patient input message"),
    files: Optional[List[UploadFile]] = None,
    model: str = Form(..., description="Backend model: 'gpt' or 'gemini'"),
    case_id: Optional[str] = Form(None, description="Existing case ID if continuing"),
    db: Session = Depends(get_db)
):
    """
    Orchestrates the patient case through the GP agent and routes based on the response.
    Maintains state across follow-ups using Postgres.
    """
    message=message.replace("'","")
    logger.info("Starting orchestrator for patient case")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    uploaded_file_paths = []
    files_content = ""
    follow_up_text = ""
    if files:
        for file in files:
            ext = file.filename.split(".")[-1].lower()
            if ext not in ["pdf", "jpg", "jpeg", "png"]:
                logger.warning(f"Skipping unsupported file type: {file.filename}")
                continue
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            if file_path in uploaded_file_paths:
                continue

            with open(file_path, "wb") as f:
                f.write(await file.read())

            uploaded_file_paths.append((file_path, ext))

    if uploaded_file_paths:
        result = await parse_endpoint(uploaded_file_paths)
        files_content += result[0]
    else:
        files_content += "None\n"

    # Handle new vs existing case
    if not case_id:
        case_id = str(uuid.uuid4())
        case = Case(
            case_id=case_id,
            user_message=message,
            stage="init",
            answered_followups=[],
            pending_questions=[],
            specialists_required=[],
            files_content=files_content,
            timestamp=datetime.utcnow(),
            consensus_winner={}
        )
        db.add(case)
        db.commit()
        db.refresh(case)
    else:
        case = db.query(Case).filter(Case.case_id == case_id).first()
        if not case:
            raise HTTPException(status_code=404, detail="Case ID not found")

        # Append answered followups as context for GP
        if case.answered_followups:
            follow_up_text = "\nPrevious Follow-ups:\n"
            for qa in case.answered_followups:
                follow_up_text += f"Q: {qa['question']}\nA: {qa['answer']}\n"

    try:
        gp_response = await GP_assess_case(
            message=message+follow_up_text,
            case_id=case_id,
            model=model,
            db=db
        )

        if gp_response.keyword == "follow_up":
            logger.info(f"GP requested follow-up questions: {gp_response.follow_up_questions}")

            if gp_response.follow_up_questions:
                # avoid duplicating same questions
                existing_qs = set(case.pending_questions)
                new_qs = [q for q in gp_response.follow_up_questions if q not in existing_qs]
                case.pending_questions = case.pending_questions + new_qs

            case.specialists_required = gp_response.specialists_required or []
            case.stage = "follow_up"

            db.commit()
            db.refresh(case)

            next_question = case.pending_questions[0] if case.pending_questions else None
            logger.info(f"Next follow-up question: {next_question}")
            return InitialOrchestratorResponse(
                case_id=case.case_id,
                stage=case.stage,
                gp_response=gp_response.response,
                next_followup=next_question,
                answered_followups=case.answered_followups,
                specialists_required=case.specialists_required
            )

        elif gp_response.keyword == "direct":
            case.stage = "direct_reply"
            db.commit()
            db.refresh(case)

            return InitialOrchestratorResponse(
                case_id=case.case_id,
                stage=case.stage,
                gp_response=gp_response.response,
                next_followup=None,
                answered_followups=case.answered_followups,
                specialists_required=None
            )

        else:
            logger.warning(f"Unknown keyword from GP: {gp_response.keyword}")
            case.stage = "unknown"
            db.commit()
            db.refresh(case)

            return InitialOrchestratorResponse(
                case_id=case.case_id,
                stage=case.stage,
                gp_response="Unrecognized GP agent response.",
                next_followup=None,
                answered_followups=case.answered_followups,
                specialists_required=case.specialists_required
            )

    except HTTPException:
        raise
    except ValueError as e:
        # Commonly raised for missing/invalid model configuration (e.g., missing API keys)
        logger.warning(f"Validation error in orchestrator: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in orchestrator: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/answer_followup", response_model=FollowUpResponse)
async def answer_followup(
    case_id: str = Form(..., description="Case ID of the ongoing consultation"),
    answer: str = Form(..., description="Patient's answer to the current follow-up question"),
    db: Session = Depends(get_db)
):
    """
    Handles patient answers to follow-up questions.
    Moves through pending questions until all are answered.
    """
    case = db.query(Case).filter(Case.case_id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case ID not found")

    if not case.pending_questions:
        return FollowUpResponse(
            case_id=case.case_id,
            stage=case.stage,
            message="No pending follow-up questions.",
            next_followup=None,
            answered_followups=case.answered_followups,
            specialists_required=case.specialists_required
        )

    current_question = case.pending_questions[0]
    remaining_questions = case.pending_questions[1:]
    case.pending_questions = remaining_questions  
    if not case.answered_followups:
        case.answered_followups = []
    case.answered_followups = case.answered_followups + [
        {"question": current_question, "answer": answer}
    ]

    db.commit()
    db.refresh(case)

    if case.pending_questions:
        next_question = case.pending_questions[0]

        return FollowUpResponse(
            case_id=case.case_id,
            stage="follow_up",
            message="Answer recorded.",
            next_followup=next_question,
            answered_followups=case.answered_followups,
            specialists_required=case.specialists_required
        )
    else:
        if case.specialists_required:
            case.stage = "initial_round"
            next_action_message = "All GP follow-up questions answered. Forwarding to specialists."
        else:
            case.stage = "completed"
            next_action_message = "All GP follow-up questions answered. Case completed."

        db.commit()
        db.refresh(case)

        return FollowUpResponse(
            case_id=case.case_id,
            stage=case.stage,
            message=next_action_message,
            next_followup=None,
            answered_followups=case.answered_followups,
            specialists_required=case.specialists_required
        )


@router.get("/get_case_state/{case_id}", response_model=FollowUpResponse)
async def get_case_state(case_id: str, db: Session = Depends(get_db)):
    """
    Get the current state of a case for debugging purposes.
    """
    case = db.query(Case).filter(Case.case_id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case ID not found")

    next_followup = case.pending_questions[0] if case.pending_questions else None

    return FollowUpResponse(
        case_id=case.case_id,
        stage=case.stage,
        message="Debug info: current case state.",
        next_followup=next_followup,
        answered_followups=case.answered_followups,
        specialists_required=case.specialists_required
    )

@router.post("/specialist_rounds")
async def specialist_rounds(
    case_id: str = Form(...),
    model: str = Form(...),
    db: Session = Depends(get_db)
):
    case = db.query(Case).filter(Case.case_id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    logger.info(f"Case {case_id} specialist rounds")
    message=db.query(Case.user_message).filter(Case.case_id == case_id).scalar()
    initial_responses=await initial_round(case_id,model,db)
    
    logger.info(f"Case {case_id} First Debate Round")
    first_debate_responses=await debate_round(case_id,model,db)
    follow_ups_common = first_debate_responses.get("follow_ups_common", (None, None))
    if follow_ups_common and follow_ups_common[1] is not None:
        logger.info(f"Case {case_id} Follow-up Specialist Rounds")
        return first_debate_responses

    logger.info(f"Case {case_id} Improved Diagnosis Round")
    improved_diagnosis_responses = await specialists_improved_diagnosis(case_id, model, db)

    logger.info(f"Case {case_id} Consensus Winner Round")
    consensus_winner_response = await determine_consensus_winner(case_id, model, db)

    return {
        **first_debate_responses,
        "improved_diagnosis": improved_diagnosis_responses,
        "consensus": consensus_winner_response
    }


def _latest_specialist_row(db: Session, case_id: str, model_cls):
    return (
        db.query(model_cls)
        .filter(model_cls.case_id == case_id)
        .order_by(model_cls.timestamp.desc())
        .first()
    )


@router.get("/get_specialist_followup_state/{case_id}", response_model=FollowUpResponseSpecialists)
async def get_specialist_followup_state(case_id: str, db: Session = Depends(get_db)):
    """
    Returns the next pending specialist follow-up question (if any).
    This is separate from GP follow-ups.
    """
    case = db.query(Case).filter(Case.case_id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case ID not found")

    specialists_required = case.specialists_required or []
    # We store the common pending questions in each specialist history row; pick the most recent.
    agent_models = {
        "Neurologist": NeurologistHistory,
        "Cardiologist": CardiologistHistory,
        "Ophthalmologist": OphthalmologistHistory,
    }

    for specialist in specialists_required:
        model_cls = agent_models.get(specialist)
        if not model_cls:
            continue
        row = _latest_specialist_row(db, case_id, model_cls)
        if row and row.pending_questions:
            next_q = row.pending_questions[0]
            return FollowUpResponseSpecialists(
                case_id=case_id,
                stage="follow_up_specialist",
                message="Pending specialist follow-up.",
                next_followup=next_q,
                answered_followups=row.answered_followups or [],
            )

    return FollowUpResponseSpecialists(
        case_id=case_id,
        stage=case.stage,
        message="No pending specialist follow-up questions.",
        next_followup=None,
        answered_followups=[],
    )


@router.post("/answer_specialist_followup", response_model=FollowUpResponseSpecialists)
async def answer_specialist_followup(
    case_id: str = Form(..., description="Case ID of the ongoing consultation"),
    answer: str = Form(..., description="Patient's answer to the current specialist follow-up question"),
    db: Session = Depends(get_db),
):
    """
    Records a patient's answer to the specialists' *common* follow-up queue.
    Once all specialist follow-ups are answered, this endpoint will automatically
    run the improved diagnosis + consensus rounds and return them in `message`.
    """
    case = db.query(Case).filter(Case.case_id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case ID not found")

    specialists_required = case.specialists_required or []
    agent_models = {
        "Neurologist": NeurologistHistory,
        "Cardiologist": CardiologistHistory,
        "Ophthalmologist": OphthalmologistHistory,
    }

    # Find any specialist row with pending questions (they should all share the same queue).
    active_row = None
    active_model = None
    for specialist in specialists_required:
        model_cls = agent_models.get(specialist)
        if not model_cls:
            continue
        row = _latest_specialist_row(db, case_id, model_cls)
        if row and row.pending_questions:
            active_row = row
            active_model = model_cls
            break

    if not active_row or not active_row.pending_questions:
        return FollowUpResponseSpecialists(
            case_id=case_id,
            stage=case.stage,
            message="No pending specialist follow-up questions.",
            next_followup=None,
            answered_followups=[],
        )

    current_question = active_row.pending_questions[0]
    remaining_questions = active_row.pending_questions[1:]
    answered = list(active_row.answered_followups or [])
    answered.append({"question": current_question, "answer": answer})

    # Apply updated queue + answers to all specialists' latest rows (keep them in sync).
    for specialist in specialists_required:
        model_cls = agent_models.get(specialist)
        if not model_cls:
            continue
        row = _latest_specialist_row(db, case_id, model_cls)
        if not row:
            continue
        row.pending_questions = remaining_questions
        row.answered_followups = answered

    db.commit()

    if remaining_questions:
        # still more specialist follow-ups to answer
        db.query(Case).filter(Case.case_id == case_id).update(
            {Case.stage: "follow_up_specialist"}, synchronize_session=False
        )
        db.commit()
        return FollowUpResponseSpecialists(
            case_id=case_id,
            stage="follow_up_specialist",
            message="Answer recorded.",
            next_followup=remaining_questions[0],
            answered_followups=answered,
        )

    # No more specialist follow-ups -> run improved diagnosis + consensus now.
    db.query(Case).filter(Case.case_id == case_id).update(
        {Case.stage: "improved_diagnosis"}, synchronize_session=False
    )
    db.commit()

    # Default to gemini if caller didn’t specify; match frontend’s default.
    model = "gemini"
    try:
        improved = await specialists_improved_diagnosis(case_id, model, db)
        consensus = await determine_consensus_winner(case_id, model, db)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed completing specialists for case {case_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed completing specialist consensus.")

    return FollowUpResponseSpecialists(
        case_id=case_id,
        stage="completed",
        message=str({"improved_diagnosis": improved, "consensus": consensus}),
        next_followup=None,
        answered_followups=answered,
    )



# -----------------------------
# Shutdown Hook
# -----------------------------
app.include_router(router)


@app.on_event("shutdown")
async def shutdown_event():
    """Delete upload folder when app exits."""
    try:
        if "reload" in sys.argv:
            return
        elif os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            logger.info(f"Upload folder {UPLOAD_FOLDER} deleted on shutdown.")
    except Exception as e:
        logger.error(f"Failed to clean upload folder: {e}")
