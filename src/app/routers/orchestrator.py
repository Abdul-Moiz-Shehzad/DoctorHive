from datetime import datetime
import logging
import os
import shutil
import sys
import uuid
from typing import Optional, List, Any, Dict
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Form, UploadFile, Depends
from fastapi.params import File
from sqlalchemy import text
from sqlalchemy.orm import Session
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from src.app.config import UPLOAD_FOLDER
from src.app.models import (
    InitialOrchestratorResponse,
    FollowUpResponse,
    Case,
    NeurologistHistory,
    CardiologistHistory,
    OphthalmologistHistory,
)
from src.app.routers.agents.GP import GP_assess_case
from src.app.routers.structures import (
    initial_round,
    debate_round,
    answer_followup as specialists_answer_followup,
    specialists_improved_diagnosis,
    determine_consensus_winner,
    chat_with_agent,
)
from src.database import Base, engine, SessionLocal
from src.utils.parse.parse_file import parse_endpoint
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
        load_dotenv()
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
            consensus_winner={},
            debate_round_count=0
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
            case.stage = "general_follow_up"

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
            stage="general_follow_up",
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

MAX_DEBATE_ROUNDS = 2

def _get_specialist_table_by_name(name: str):
    mapping = {
        "Neurologist": NeurologistHistory,
        "Cardiologist": CardiologistHistory,
        "Ophthalmologist": OphthalmologistHistory,
    }
    return mapping.get(name)

def _resolve_specialist_check_flag(case: Case, db: Session) -> Optional[str]:
    """
    specialists.py answer_followup requires check_flag.
    To reduce frontend load, infer it here.
    Since debate_round writes the same common followups to all specialist tables,
    we can use the first specialist that still has pending questions.
    """
    specialists_required = case.specialists_required or []

    for specialist in specialists_required:
        table = _get_specialist_table_by_name(specialist)
        if not table:
            continue

        latest = (
            db.query(table)
            .filter(table.case_id == case.case_id)
            .order_by(table.timestamp.desc())
            .first()
        )

        if latest and latest.pending_questions:
            return specialist

    return specialists_required[0] if specialists_required else None

@router.post("/doctorhive")
async def doctorhive(
    case_id: Optional[str] = Form(
        None,
        description="Case ID for an existing consultation. Leave empty to start a new case."
    ),
    model: str = Form(
        ...,
        description="LLM model to use for processing (e.g., 'gemini')."
    ),
    message: Optional[str] = Form(
        None,
        description="Patient's main message or complaint. Required when starting a new case or sending a new query."
    ),
    answer: Optional[str] = Form(
        None,
        description="Patient's answer to a follow-up question (GP or specialist)."
    ),
    files: Optional[List[UploadFile]] = File(
        None,
        description="Optional medical files (reports, images, etc.) uploaded by the patient."
    ),
    agent_name: Optional[str] = Form(
        None,
        description="Name of the specialist to chat with after consensus (e.g., Neurologist). Used only in transfer_control stage."
    ),
    chat_type: Optional[int] = Form(
        None,
        description="Type of interaction in transfer_control stage: 0 = get recommendation/summary from winning specialist, 1 = direct chat with specialist."
    ),
    user_message: Optional[str] = Form(
        None,
        description="Message sent by the user when chatting directly with a specialist (chat_type=1)."
    ),
    consensus_data_json: Optional[str] = Form(
        None,
        description="Optional JSON string containing consensus or decision data passed to the final stage."
    ),
    db: Session = Depends(get_db),
):
    """
    Single driver endpoint for the full lifecycle.

    Rules:
    - no case_id => create/process new case via GP orchestrator
    - existing case => inspect case.stage and automatically call the correct function
    """

    # normalize swagger empty strings
    case_id = case_id.strip() if case_id else None
    message = message.strip() if message else None
    answer = answer.strip() if answer else None
    agent_name = agent_name.strip() if agent_name else None
    user_message = user_message.strip() if user_message else None
    consensus_data_json = consensus_data_json.strip() if consensus_data_json else None
    files = files or []

    # -----------------------------
    # 1) New case: start from GP orchestration
    # -----------------------------
    if not case_id:
        if not message:
            raise HTTPException(
                status_code=400,
                detail="message is required when case_id is not provided"
            )

        return await process_patient_message_and_files(
            message=message,
            files=files,
            model=model,
            case_id=None,
            db=db,
        )

    # -----------------------------
    # 2) Existing case
    # -----------------------------
    case = db.query(Case).filter(Case.case_id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case ID not found")

    stage = case.stage

    # -----------------------------
    # GP stages
    # -----------------------------
    if stage == "init":
        if not message:
            raise HTTPException(
                status_code=400,
                detail="message is required for stage='init'"
            )

        return await process_patient_message_and_files(
            message=message,
            files=files,
            model=model,
            case_id=case_id,
            db=db,
        )

    if stage == "general_follow_up":
        if not answer:
            next_question = case.pending_questions[0] if case.pending_questions else None
            return {
                "case_id": case.case_id,
                "stage": case.stage,
                "message": "Patient answer required for GP follow-up.",
                "next_followup": next_question,
                "answered_followups": case.answered_followups,
                "specialists_required": case.specialists_required,
            }

        return await answer_followup(
            case_id=case_id,
            answer=answer,
            db=db,
        )

    if stage == "direct_reply":
        return {
            "case_id": case.case_id,
            "stage": stage,
            "message": "GP has already provided a direct reply for this case.",
            "next_action": "frontend can show GP response and optionally close this case",
        }

    if stage == "unknown":
        return {
            "case_id": case.case_id,
            "stage": stage,
            "message": "Case is in unknown state. Manual inspection required.",
        }

    # -----------------------------
    # Specialist stages
    # -----------------------------
    if stage == "initial_round":
        result = await initial_round(
            case_id=case_id,
            model=model,
            db=db,
        )

        refreshed_case = db.query(Case).filter(Case.case_id == case_id).first()
        refreshed_case.debate_round_count = 1
        db.commit()
        db.refresh(refreshed_case)

        return {
            "case_id": case_id,
            "stage_before": "initial_round",
            "stage_after": refreshed_case.stage,
            "debate_round_count": refreshed_case.debate_round_count,
            "data": result,
        }

    if stage == "debate":
        result = await debate_round(
            case_id=case_id,
            model=model,
            db=db,
        )

        refreshed_case = db.query(Case).filter(Case.case_id == case_id).first()
        refreshed_case.debate_round_count += 1
        db.commit()
        db.refresh(refreshed_case)

        return {
            "case_id": case_id,
            "stage_before": "debate",
            "stage_after": refreshed_case.stage,
            "debate_round_count": refreshed_case.debate_round_count,
            "data": result,
        }

    if stage == "specialists_follow_up":
        check_flag = _resolve_specialist_check_flag(case, db)

        if not answer:
            specialist_table = _get_specialist_table_by_name(check_flag) if check_flag else None
            latest = None

            if specialist_table:
                latest = (
                    db.query(specialist_table)
                    .filter(specialist_table.case_id == case_id)
                    .order_by(specialist_table.timestamp.desc())
                    .first()
                )

            next_question = latest.pending_questions[0] if latest and latest.pending_questions else None

            return {
                "case_id": case.case_id,
                "stage": stage,
                "message": "Patient answer required for specialist follow-up.",
                "check_flag": check_flag,
                "next_followup": next_question,
            }

        if not check_flag:
            raise HTTPException(
                status_code=500,
                detail="Could not infer specialist follow-up owner"
            )

        return await specialists_answer_followup(
            case_id=case_id,
            answer=answer,
            db=db,
            check_flag=check_flag,
        )

    if stage == "improved_diagnosis":
        result = await specialists_improved_diagnosis(
            case_id=case_id,
            model=model,
            db=db,
        )

        refreshed_case = db.query(Case).filter(Case.case_id == case_id).first()
        return {
            "case_id": case_id,
            "stage_before": "improved_diagnosis",
            "stage_after": refreshed_case.stage,
            "debate_round_count": refreshed_case.debate_round_count,
            "data": result,
        }

    if stage == "choice":
        if case.debate_round_count < MAX_DEBATE_ROUNDS:
            db.query(Case).filter(Case.case_id == case_id).update(
                {Case.stage: "debate"},
                synchronize_session=False,
            )
            db.commit()

            result = await debate_round(
                case_id=case_id,
                model=model,
                db=db,
            )

            refreshed_case = db.query(Case).filter(Case.case_id == case_id).first()
            refreshed_case.debate_round_count += 1
            db.commit()
            db.refresh(refreshed_case)

            return {
                "case_id": case_id,
                "decision": "loop_back_to_debate",
                "debate_round_count": refreshed_case.debate_round_count,
                "stage_after": refreshed_case.stage,
                "data": result,
            }

        consensus_result = await determine_consensus_winner(
            case_id=case_id,
            model=model,
            db=db,
        )

        db.query(Case).filter(Case.case_id == case_id).update(
            {
                Case.consensus_winner: consensus_result,
                Case.stage: "transfer_control",
            },
            synchronize_session=False,
        )
        db.commit()

        refreshed_case = db.query(Case).filter(Case.case_id == case_id).first()

        return {
            "case_id": case_id,
            "decision": "move_to_consensus",
            "debate_round_count": refreshed_case.debate_round_count,
            "stage_after": refreshed_case.stage,
            "data": consensus_result,
        }

    if stage == "transfer_control":
        resolved_chat_type = 0 if chat_type is None else chat_type

        if resolved_chat_type == 0:
            winner_data = case.consensus_winner or {}

            if not winner_data:
                winner_data = await determine_consensus_winner(
                    case_id=case_id,
                    model=model,
                    db=db,
                )

                db.query(Case).filter(Case.case_id == case_id).update(
                    {Case.consensus_winner: winner_data},
                    synchronize_session=False,
                )
                db.commit()

            winner_name = winner_data.get("winner") if isinstance(winner_data, dict) else None
            if not winner_name:
                raise HTTPException(
                    status_code=500,
                    detail="Consensus winner not available for transfer_control stage"
                )
            result = await chat_with_agent(
                case_id=case_id,
                model=model,
                agent_name=winner_name,
                chat_type=0,
                user_message=None,
                consensus_data_json=consensus_data_json,
                db=db,
            )
            refreshed_case = db.query(Case).filter(Case.case_id == case_id).first()

            previous_text=refreshed_case.consensus_winner
            diagnosis=previous_text.get("diagnosis")
            explanation=previous_text.get("explanation")

            message = result.get("message")
            winner = result.get("agent_name")

            refreshed_case.consensus_winner = {"winner": winner, "message":message, "diagnosis":diagnosis, "explanation":explanation}
            refreshed_case.stage = "completed"
            db.commit()
            return result

        if resolved_chat_type == 1:
            if not agent_name:
                winner_data = case.consensus_winner or {}
                if isinstance(winner_data, dict):
                    agent_name = winner_data.get("winner")

            if not agent_name:
                raise HTTPException(
                    status_code=400,
                    detail="agent_name is required for chat_type=1"
                )

            if not user_message:
                raise HTTPException(
                    status_code=400,
                    detail="user_message is required for chat_type=1"
                )

            return await chat_with_agent(
                case_id=case_id,
                model=model,
                agent_name=agent_name,
                chat_type=1,
                user_message=user_message,
                consensus_data_json=consensus_data_json,
                db=db,
            )

        raise HTTPException(status_code=400, detail="Invalid chat_type")

    if stage == "completed":
        return {
            "case_id": case.case_id,
            "stage": stage,
            "message": "Case already completed.",
            "consensus_winner": case.consensus_winner,
        }

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported stage '{stage}'"
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
