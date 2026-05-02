from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.app.models import PatientProfile
from src.utils.utilities import get_db

router = APIRouter(prefix="/profile", tags=["profile"])


class ProfileSaveRequest(BaseModel):
    user_id: int
    age: Optional[int] = None
    gender: Optional[str] = None
    blood_type: Optional[str] = None
    allergies: Optional[List[str]] = []
    conditions: Optional[List[str]] = []
    medications: Optional[List[str]] = []
    smoking: Optional[str] = None
    alcohol: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None


@router.post("/save")
async def save_profile(req: ProfileSaveRequest, db: Session = Depends(get_db)):
    """Create or update a patient's medical profile."""
    existing = db.query(PatientProfile).filter(PatientProfile.user_id == req.user_id).first()
    if existing:
        existing.age = req.age
        existing.gender = req.gender
        existing.blood_type = req.blood_type
        existing.allergies = req.allergies or []
        existing.conditions = req.conditions or []
        existing.medications = req.medications or []
        existing.smoking = req.smoking
        existing.alcohol = req.alcohol
        existing.emergency_contact_name = req.emergency_contact_name
        existing.emergency_contact_phone = req.emergency_contact_phone
        existing.updated_at = datetime.utcnow()
    else:
        db.add(PatientProfile(
            user_id=req.user_id,
            age=req.age,
            gender=req.gender,
            blood_type=req.blood_type,
            allergies=req.allergies or [],
            conditions=req.conditions or [],
            medications=req.medications or [],
            smoking=req.smoking,
            alcohol=req.alcohol,
            emergency_contact_name=req.emergency_contact_name,
            emergency_contact_phone=req.emergency_contact_phone,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ))
    db.commit()
    return {"status": "saved"}


@router.get("/{user_id}")
async def get_profile(user_id: int, db: Session = Depends(get_db)):
    """Get a user's medical profile. Returns 404 if not yet created."""
    p = db.query(PatientProfile).filter(PatientProfile.user_id == user_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {
        "user_id": p.user_id,
        "age": p.age,
        "gender": p.gender,
        "blood_type": p.blood_type,
        "allergies": p.allergies or [],
        "conditions": p.conditions or [],
        "medications": p.medications or [],
        "smoking": p.smoking,
        "alcohol": p.alcohol,
        "emergency_contact_name": p.emergency_contact_name,
        "emergency_contact_phone": p.emergency_contact_phone,
    }
