from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    conversation_id: str
    student_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    agent_used: str
    actions_taken: list[str]
    student_summary: dict | None = None
