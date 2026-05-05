"""Minimal A2A HTTP+JSON server for the engineer agent.

Implements key A2A endpoints from the spec:
- GET /.well-known/agent-card.json
- POST /message:send
- GET /tasks/{id}

This keeps the implementation simple while making the agent truly callable via
A2A-style protocol operations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .main import run_engineer_agent


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class A2APart(BaseModel):
    text: Optional[str] = None
    raw: Optional[str] = None
    url: Optional[str] = None
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    filename: Optional[str] = None
    mediaType: Optional[str] = None


class A2AMessage(BaseModel):
    messageId: str = Field(default_factory=lambda: str(uuid4()))
    contextId: Optional[str] = None
    taskId: Optional[str] = None
    role: Literal["ROLE_USER", "ROLE_AGENT"]
    parts: List[A2APart]
    metadata: Optional[Dict[str, Any]] = None
    extensions: Optional[List[str]] = None
    referenceTaskIds: Optional[List[str]] = None


class SendMessageConfiguration(BaseModel):
    acceptedOutputModes: Optional[List[str]] = None
    historyLength: Optional[int] = None
    returnImmediately: Optional[bool] = None


class SendMessageRequest(BaseModel):
    message: A2AMessage
    configuration: Optional[SendMessageConfiguration] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskStatus(BaseModel):
    state: str
    message: Optional[A2AMessage] = None
    timestamp: Optional[str] = None


class Task(BaseModel):
    id: str
    contextId: Optional[str] = None
    status: TaskStatus
    artifacts: Optional[List[Dict[str, Any]]] = None
    history: Optional[List[A2AMessage]] = None
    metadata: Optional[Dict[str, Any]] = None


class SendMessageResponse(BaseModel):
    task: Optional[Task] = None
    message: Optional[A2AMessage] = None


class EngineerA2AServer:
    """Simple in-memory A2A server facade for the engineer agent."""

    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}

    def get_agent_card(self, base_url: str = "http://localhost:8102") -> Dict[str, Any]:
        return {
            "name": "Engineer Resume Agent",
            "description": "Skeptical technical interviewer agent for resume evaluation.",
            "version": "1.0.0",
            "capabilities": {
                "streaming": False,
                "pushNotifications": False,
                "extendedAgentCard": False,
            },
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["application/json"],
            "supportedInterfaces": [
                {
                    "url": base_url,
                    "protocolBinding": "HTTP+JSON",
                    "protocolVersion": "1.0",
                }
            ],
            "skills": [
                {
                    "id": "technical_resume_review",
                    "name": "Technical Resume Review",
                    "description": "Scores technical interview-slot worthiness with concrete criticisms and one strength.",
                    "tags": ["resume", "engineering", "technical-screen"],
                    "examples": [
                        "Evaluate this software engineer resume for technical depth.",
                    ],
                    "inputModes": ["text/plain"],
                    "outputModes": ["application/json"],
                }
            ],
        }

    def send_message(self, payload: SendMessageRequest) -> SendMessageResponse:
        if payload.message.role != "ROLE_USER":
            raise HTTPException(status_code=400, detail="Only ROLE_USER input is supported.")

        text_parts = [part.text for part in payload.message.parts if part.text]
        if not text_parts:
            raise HTTPException(status_code=400, detail="At least one text part is required.")

        resume_text = "\n".join(text_parts).strip()
        if not resume_text:
            raise HTTPException(status_code=400, detail="Resume text cannot be empty.")

        artifact = run_engineer_agent(resume_text)

        task_id = payload.message.taskId or str(uuid4())
        context_id = payload.message.contextId or str(uuid4())
        status = TaskStatus(
            state="TASK_STATE_COMPLETED",
            timestamp=_utc_now_iso(),
        )
        task = Task(
            id=task_id,
            contextId=context_id,
            status=status,
            artifacts=[artifact],
            history=[payload.message],
            metadata={"agent": "engineer_agent"},
        )
        self._tasks[task_id] = task
        return SendMessageResponse(task=task)

    def get_task(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found.")
        return task


def create_app(base_url: str = "http://localhost:8102") -> FastAPI:
    server = EngineerA2AServer()
    app = FastAPI(title="Engineer Agent A2A Server", version="1.0.0")

    @app.get("/.well-known/agent-card.json")
    def agent_card() -> Dict[str, Any]:
        return server.get_agent_card(base_url=base_url)

    @app.post("/message:send", response_model=SendMessageResponse)
    def send_message(
        payload: SendMessageRequest,
        a2a_version: Optional[str] = Header(default="1.0", alias="A2A-Version"),
    ) -> SendMessageResponse:
        if a2a_version and a2a_version not in {"1.0", "0.3"}:
            raise HTTPException(status_code=400, detail="VersionNotSupportedError")
        return server.send_message(payload)

    @app.get("/tasks/{task_id}", response_model=Task)
    def get_task(task_id: str) -> Task:
        return server.get_task(task_id)

    return app


app = create_app()
