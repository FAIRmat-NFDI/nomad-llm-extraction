from pydantic import BaseModel

class SimpleWorkflowInput(BaseModel):
    """Input model for the simple Temporal workflow"""
    upload_id: str
    user_id: str