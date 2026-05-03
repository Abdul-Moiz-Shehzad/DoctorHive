
from datetime import datetime
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from src.database import Base, engine, SessionLocal
from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

class UserInput(BaseModel):
    """Model for capturing patient input sent to the orchestrator."""
    message: str
    files: Optional[List[str]] = None


class GPResponse(BaseModel):
    """Model for the response returned by the GP agent."""
    keyword: str
    response: str
    follow_up_questions: Optional[List[str]]
    specialists_required: Optional[List[str]]

class InitialOrchestratorResponse(BaseModel):
    """Response for the initial GP assessment."""
    case_id: str
    stage: str
    gp_response: str
    next_followup: Optional[str] = None
    answered_followups: List[Dict[str, Any]] = []
    specialists_required: Optional[List[str]] = None
    chat_name: str

class FollowUpResponse(BaseModel):
    """Response for follow-up questions after initial GP response."""
    case_id: str
    stage: str
    next_followup: Optional[str] = None
    message: Optional[str] = None
    answered_followups: List[Dict[str, Any]] = []
    specialists_required: Optional[List[str]] = None

class FollowUpResponseSpecialists(BaseModel):
    """Response for follow-up questions for specialists."""
    case_id: str
    stage: str
    next_followup: Optional[str] = None
    message: Optional[str] = None
    answered_followups: List[Dict[str, Any]] = []


class OrchestratorResponse(BaseModel):
    """Deprecated- use instead InitialOrchestratorResponse or FollowUpResponse."""
    case_id: str 
    stage: str  
    message: str  
    next_action: Optional[str] = None  
    follow_up_questions: Optional[List[str]] = None 
    answered_followups: Optional[List[Dict[str, Any]]] = None  
    specialists_required: Optional[List[str]] = None  

class Case(Base):
    __tablename__ = "cases"

    case_id = Column(String, primary_key=True, index=True)
    user_message = Column(String)
    stage = Column(String, default="init")
    answered_followups = Column(JSONB, default=list)
    pending_questions = Column(JSONB, default=list)
    specialists_required = Column(JSONB, default=list)
    files_content = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    consensus_winner = Column(JSONB, default=dict)
    debate_round_count = Column(Integer, default=0)
    chat_name = Column(String, nullable=True)

class Specialized_Agents_Diagnosis_Response(BaseModel):
    """Response for the Specialized agents diagnosis."""
    confidence: int
    diagnosis: str
    explanation: str


class NeurologistHistory(Base):
    __tablename__ = "neurologist_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    case_id = Column(String, ForeignKey("cases.case_id"))
    user_input = Column(String)
    agent_response = Column(JSON)
    answered_followups = Column(JSONB, default=list)
    pending_questions = Column(JSONB, default=list)
    timestamp = Column(DateTime, default=datetime.utcnow)

class CardiologistHistory(Base):
    __tablename__ = "cardiologist_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    case_id = Column(String, ForeignKey("cases.case_id"))
    user_input = Column(String)
    agent_response = Column(JSON)
    answered_followups = Column(JSONB, default=list)
    pending_questions = Column(JSONB, default=list)
    timestamp = Column(DateTime, default=datetime.utcnow)

class OphthalmologistHistory(Base):
    __tablename__ = "ophthalmologist_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    case_id = Column(String, ForeignKey("cases.case_id"))
    user_input = Column(String)
    agent_response = Column(JSON)
    answered_followups = Column(JSONB, default=list)
    pending_questions = Column(JSONB, default=list)
    timestamp = Column(DateTime, default=datetime.utcnow)


# ─── Auth & Frontend State Tables ────────────────────────────────────────────

class User(Base):
    """Registered users with hashed passwords."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    preferred_model = Column(String, default="gemini")
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatHistory(Base):
    """Snapshot of a frontend consultation session for a given user+case."""
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    case_id = Column(String, ForeignKey("cases.case_id"), nullable=False)
    # JSONB snapshot: {gp_response, stage, specialists_required, answered_followups, specialist_result, submitted_message}
    snapshot = Column(JSONB, nullable=False, default=dict)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserCaseMapping(Base):
    """Maps which user owns which case."""
    __tablename__ = "user_case_mapping"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    case_id = Column(String, ForeignKey("cases.case_id"), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PatientProfile(Base):
    """Medical profile for a registered user — filled on first login, editable anytime."""
    __tablename__ = "patient_profiles"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)           # Male / Female / Other / Prefer not to say
    blood_type = Column(String, nullable=True)        # A+, B-, O+, AB+, etc.
    allergies = Column(JSONB, default=list)           # ["Penicillin", "Pollen"]
    conditions = Column(JSONB, default=list)          # ["Diabetes Type 2", "Hypertension"]
    medications = Column(JSONB, default=list)         # ["Metformin 500mg"]
    smoking = Column(String, nullable=True)           # Never / Occasionally / Regularly
    alcohol = Column(String, nullable=True)           # Never / Occasionally / Regularly
    emergency_contact_name = Column(String, nullable=True)
    emergency_contact_phone = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


