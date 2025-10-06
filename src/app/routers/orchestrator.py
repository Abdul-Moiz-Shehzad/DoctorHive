import logging
import os
import shutil
import sys
import uuid
from app.routers.structures import first_debate_round, initial_round
from fastapi import APIRouter, FastAPI, HTTPException, Form, UploadFile, Depends
from typing import Optional, List
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.app.models import InitialOrchestratorResponse, FollowUpResponse, Case
from src.app.routers.agents.GP import GP_assess_case
from src.utils.parse.parse_file import parse_endpoint
from src.app.config import UPLOAD_FOLDER
from src.database import Base, engine, SessionLocal
from src.utils.utilities import get_db

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
            files_content=files_content
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

    except Exception as e:
        logger.error(f"Error in orchestrator: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    initial_responses=await initial_round(case_id,message,model,db)
    db.query(Case).filter(Case.case_id == case_id).update(
        {Case.stage: "first_debate"}, synchronize_session=False
    )
    db.commit()
    first_debate_responses=await first_debate_round(case_id,model,initial_responses["responses"],db)
    return initial_responses



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
