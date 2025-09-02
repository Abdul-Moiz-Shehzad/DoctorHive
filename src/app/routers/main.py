import logging
import os
import shutil
import sys
from fastapi import APIRouter, FastAPI, HTTPException, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List

from fastapi.responses import RedirectResponse
from src.app.routers.agents import GP
from src.app.routers import orchestrator
from src.utils import heartbeat
from src.app.config import UPLOAD_FOLDER

app = FastAPI(
    title="Main",
    description="All routers endpoint",
    version="1.0"
    )
router = APIRouter(prefix="/main", tags=["main"])
logger = logging.getLogger(__name__)

app.include_router(router)



# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(GP.router, tags=["General Physician"])
app.include_router(orchestrator.router, tags=["orchestrator"])
app.include_router(heartbeat.router, tags=["heartbeat"])

@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")

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