import logging
import os
import shutil
import sys
from fastapi import APIRouter, FastAPI, HTTPException, Form, UploadFile
from typing import Optional, List
from src.app.models import OrchestratorResponse
from src.app.routers.agents.GP import GP_assess_case
from src.utils.parse.parse_file import parse_endpoint
from src.app.config import UPLOAD_FOLDER

app = FastAPI(
    title="Orchestrator",
    description="Handles all the flow",
    version="1.0"
    )
router = APIRouter(prefix="/orchestrator", tags=["Orchestrator"])
logger = logging.getLogger(__name__)


@app.post("/process")#, response_model=OrchestratorResponse)
async def orchestrate_case(
    message: str = Form(..., description="Patient input message"),
    files: Optional[List[UploadFile]] = None,
    model: str = Form(..., description="Backend model: 'gpt' or 'gemini'")
):
    """
    Orchestrates the patient case through the GP agent and routes based on the response.
    
    User Input:
    - Message: {message}
    - Files: {files_content}
    - Model: {model}
    """
    logger.info("Starting orchestrator for patient case")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info(f"Upload folder ensured at {UPLOAD_FOLDER}")
    uploaded_file_paths = []
    files_content=None
    if files:
        for file in files:
            ext = file.filename.split(".")[-1].lower()
            if ext not in ["pdf", "jpg", "jpeg", "png"]:
                logger.warning(f"Skipping unsupported file type: {file.filename}")
                continue
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            if file_path in uploaded_file_paths:
                logger.info(f"File already uploaded, skipping: {file.filename}")
                continue

            with open(file_path, "wb") as f:
                f.write(await file.read())

            uploaded_file_paths.append((file_path,ext))
            logger.info(f"Uploaded file saved: {file_path}")

    if uploaded_file_paths:
        logger.info(f"Parsing {len(uploaded_file_paths)} uploaded files")
        files_content = await parse_endpoint(uploaded_file_paths)
    else:
        files_content = ""
    return files_content
    try:
        gp_response = await GP_assess_case(
            message=message,
            files_content=files_content,
            model=model
        )

        if gp_response.keyword == "follow-up-questions-gp":
            return OrchestratorResponse(
                stage="follow_up",
                message=gp_response.response,
                next_action="ask_follow_up",
                follow_up_questions=gp_response.follow_up_questions,
                specialists_required=gp_response.specialists_required
            )

        elif gp_response.keyword == "direct":
            return OrchestratorResponse(
                stage="direct_reply",
                message=gp_response.response,
                next_action="end",
                follow_up_questions=None,
                specialists_required=None
            )

        else:
            logger.warning(f"Unknown keyword from GP: {gp_response.keyword}")
            return OrchestratorResponse(
                stage="unknown",
                message="Unrecognized GP agent response.",
                next_action="manual_review"
            )

    except Exception as e:
        logger.error(f"Error in orchestrator: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router)
@app.on_event("shutdown")
async def shutdown_event():
    """Delete upload folder when app exits."""
    try:
        if "reload" in sys.argv:
            logger.info("Skipping cleanup during autoreload.")
            return
        elif os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            logger.info(f"Upload folder {UPLOAD_FOLDER} deleted on shutdown.")
    except Exception as e:
        logger.error(f"Failed to clean upload folder: {e}")