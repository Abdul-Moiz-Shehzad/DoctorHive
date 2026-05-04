import logging
import os
import shutil
import sys
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from typing import Optional, List

from fastapi.responses import RedirectResponse
from src.app.routers.agents import GP
from src.app.routers import structures
from src.app.routers import orchestrator
from src.app.routers import auth as auth_router
from src.app.routers import patient_profile as profile_router
from src.utils import heartbeat
from src.app.config import UPLOAD_FOLDER
from src.database import engine

load_dotenv()

app = FastAPI(
    title="Main",
    description="All routers endpoint",
    version="1.0"
    )
router = APIRouter(prefix="/main", tags=["main"])
logger = logging.getLogger(__name__)

app.include_router(router)

BACKEND_FRONTEND_TOKEN = os.getenv("BACKEND_FRONTEND_TOKEN")

ALLOWED_PUBLIC_PATHS = {
    "/",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/favicon.ico",
}
AUTH_EXEMPT_PATHS = {
    "/auth/register",
    "/auth/login",
}

@app.middleware("http")
async def validate_backend_frontend_token(request: Request, call_next):
    # Allow CORS preflight and public auth endpoints without the hidden token.
    if request.method == "OPTIONS" or request.url.path in ALLOWED_PUBLIC_PATHS or request.url.path in AUTH_EXEMPT_PATHS:
        return await call_next(request)

    header_token = request.headers.get("x-backend-token")
    if not BACKEND_FRONTEND_TOKEN or header_token != BACKEND_FRONTEND_TOKEN:
        raise HTTPException(status_code=403, detail="Hidden backend token missing or invalid.")
    return await call_next(request)


@app.on_event("startup")
async def ensure_user_schema():
    with engine.connect() as conn:
        conn.execute(text(
            "ALTER TABLE IF EXISTS users "
            "ADD COLUMN IF NOT EXISTS preferred_model VARCHAR DEFAULT 'gemini', "
            "ADD COLUMN IF NOT EXISTS preferred_theme VARCHAR DEFAULT 'dark';"
        ))
        conn.commit()


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(orchestrator.router, tags=["orchestrator"])
app.include_router(auth_router.router, tags=["auth"])
app.include_router(profile_router.router, tags=["profile"])
app.include_router(structures.router, tags=["structures"])
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