
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

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

class OrchestratorResponse(BaseModel):
    """Final response returned by the orchestrator to the client."""
    stage: str
    message: str
    next_action: Optional[str] = None
    follow_up_questions: Optional[List[str]] = None
    specialists_required: Optional[List[str]] = None