
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

class FollowUpResponse(BaseModel):
    """Response for follow-up questions after initial GP response."""
    case_id: str
    stage: str
    next_followup: Optional[str] = None
    message: Optional[str] = None
    answered_followups: List[Dict[str, Any]] = []
    specialists_required: Optional[List[str]] = None

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
    stage = Column(String, default="init")
    answered_followups = Column(JSONB, default=list)
    pending_questions = Column(JSONB, default=list)
    specialists_required = Column(JSONB, default=list)
    files_content = Column(String)

class Specialized_Agents_Diagnosis_Response(BaseModel):
    """Response for the initial GP assessment."""
    confidence: int
    diagnosis: str
    explanation: str


class NeurologistHistory(Base):
    __tablename__ = "neurologist_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    case_id = Column(String, ForeignKey("cases.case_id"))
    user_input = Column(String)
    agent_response = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

class CardiologistHistory(Base):
    __tablename__ = "cardiologist_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    case_id = Column(String, ForeignKey("cases.case_id"))
    user_input = Column(String)
    agent_response = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

class OphthalmologistHistory(Base):
    __tablename__ = "ophthalmologist_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    case_id = Column(String, ForeignKey("cases.case_id"))
    user_input = Column(String)
    agent_response = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
