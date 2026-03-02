"""
SmartMentor AI — FastAPI Entry Point

POST /chat with { question, conversation_id, student_id? }
Returns { response, agent_used, actions_taken, student_summary? }
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
from logger import setup_logger

logger = setup_logger("API")
from models import ChatRequest, ChatResponse
from orchestrator import run_agent
from config import GOOGLE_API_KEY


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("SmartMentor AI Starting Up...")
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not set! System will fail on LLM calls.")
    else:
        logger.info("GOOGLE_API_KEY detected.")
    logger.info("API docs available at: http://localhost:8000/docs")
    yield
    # Shutdown
    logger.info("SmartMentor AI Shutting Down.")


app = FastAPI(
    title="SmartMentor AI",
    description="AI-powered student mentoring system with Mentor, Placement, and Emotional agents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start_time = time.time()
    logger.info(f"Incoming request | Conv: {request.conversation_id} | Student: {request.student_id}")
    try:
        result = await run_agent(
            question=request.question,
            conversation_id=request.conversation_id,
            student_id=request.student_id,
        )
        duration = time.time() - start_time
        logger.info(f"Request completed in {duration:.2f}s | Agent: {result['agent_used']}")
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.get("/")
async def root():
    return {
        "service": "SmartMentor AI",
        "version": "1.0.0",
        "endpoints": {
            "POST /chat": "Main chat endpoint — send question + conversation_id",
            "GET /docs": "Interactive API documentation (Swagger)",
        },
        "sample_payload": {
            "question": "How is student S001 doing?",
            "conversation_id": "conv-1",
            "student_id": "S001",
        },
    }
