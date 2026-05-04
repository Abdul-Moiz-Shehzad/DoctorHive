import os
import bcrypt
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.app.models import (
    User,
    Case,
    ChatHistory,
    UserCaseMapping,
    PatientProfile,
    NeurologistHistory,
    CardiologistHistory,
    OphthalmologistHistory,
)
from src.utils.utilities import get_db

load_dotenv()

# ─── Config ──────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET", "doctorhive-change-this-secret-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30  # Long-lived — user stays logged in

router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


# ─── Schemas ──────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str
    email: str
    preferred_model: str
    preferred_theme: str

class UserResponse(BaseModel):
    user_id: int
    username: str
    email: str
    preferred_model: str
    preferred_theme: str


# ─── Helpers ─────────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    """Hash a password using bcrypt directly (compatible with bcrypt >= 4.0)."""
    # Truncate to 72 bytes for bcrypt compatibility
    password_bytes = password[:72].encode("utf-8")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password_bytes, salt).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    # Truncate to 72 bytes for bcrypt compatibility
    return bcrypt.checkpw(plain[:72].encode("utf-8"), hashed.encode("utf-8"))

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> Optional[User]:
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            return None
    except JWTError:
        return None
    return db.query(User).filter(User.id == int(user_id)).first()


# ─── Endpoints ───────────────────────────────────────────────────────────────
@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user. Returns a JWT immediately so the user is logged in."""
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        username=req.username,
        email=req.email,
        hashed_password=hash_password(req.password),
        created_at=datetime.utcnow(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": str(user.id)})
    return TokenResponse(
        access_token=token, 
        user_id=user.id, 
        username=user.username, 
        email=user.email,
        preferred_model=user.preferred_model or "gemini",
        preferred_theme=user.preferred_theme or "dark"
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    """Login with username + password. Returns JWT."""
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"sub": str(user.id)})
    return TokenResponse(
        access_token=token, 
        user_id=user.id, 
        username=user.username, 
        email=user.email,
        preferred_model=user.preferred_model or "gemini",
        preferred_theme=user.preferred_theme or "dark"
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: Optional[User] = Depends(get_current_user)):
    """Validate a JWT and return the current user's info."""
    return UserResponse(
        user_id=current_user.id, 
        username=current_user.username, 
        email=current_user.email,
        preferred_model=current_user.preferred_model or "gemini",
        preferred_theme=current_user.preferred_theme or "dark"
    )


class UpdateModelRequest(BaseModel):
    model: str

class UpdateThemeRequest(BaseModel):
    theme: str

@router.post("/update-model")
async def update_model(req: UpdateModelRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update the user's preferred model."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    current_user.preferred_model = req.model
    db.commit()
    return {"status": "model updated", "preferred_model": current_user.preferred_model}


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

@router.post("/change-password")
async def change_password(req: ChangePasswordRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Verify current password and update to new password."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    if not verify_password(req.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect current password")
    
    current_user.hashed_password = hash_password(req.new_password)
    db.commit()
    return {"status": "password updated"}

@router.delete('/delete-account')
async def delete_account(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Delete the current user and any owned consultation history."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    case_ids = [m.case_id for m in db.query(UserCaseMapping).filter(UserCaseMapping.user_id == current_user.id).all()]

    if case_ids:
        db.query(NeurologistHistory).filter(NeurologistHistory.case_id.in_(case_ids)).delete(synchronize_session=False)
        db.query(CardiologistHistory).filter(CardiologistHistory.case_id.in_(case_ids)).delete(synchronize_session=False)
        db.query(OphthalmologistHistory).filter(OphthalmologistHistory.case_id.in_(case_ids)).delete(synchronize_session=False)
        db.query(ChatHistory).filter(ChatHistory.case_id.in_(case_ids)).delete(synchronize_session=False)
        db.query(Case).filter(Case.case_id.in_(case_ids)).delete(synchronize_session=False)
        db.query(UserCaseMapping).filter(UserCaseMapping.case_id.in_(case_ids)).delete(synchronize_session=False)

    db.query(ChatHistory).filter(ChatHistory.user_id == current_user.id).delete(synchronize_session=False)
    db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).delete(synchronize_session=False)
    db.query(UserCaseMapping).filter(UserCaseMapping.user_id == current_user.id).delete(synchronize_session=False)
    db.query(User).filter(User.id == current_user.id).delete(synchronize_session=False)
    db.commit()

    return {"status": "account deleted"}
