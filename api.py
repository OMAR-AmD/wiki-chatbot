from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime
from chatbot import WikiChatbot
from chat_storage import ChatStorage
import json
import logging
import time
from chromadb import Client
from chromadb.config import Settings

# Disable all telemetry
client = Client(Settings(
    anonymized_telemetry=False
))


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Wiki Chatbot API",
    description="Local AI Chatbot API powered by Llama 2",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
chatbot = WikiChatbot()
storage = ChatStorage()

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    type: str  # 'hybrid_rag' ou 'general_knowledge'
    latency_ms: float

class SessionInfo(BaseModel):
    session_id: str
    messages: int
    created: str

class StatisticsResponse(BaseModel):
    total_sessions: int
    total_messages: int
    avg_messages_per_session: float

# API Endpoints

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "service": "Wiki Chatbot API",
        "version": "1.0.0"
    }

@app.post("/query", response_model=QueryResponse)
async def query_chatbot(request: QueryRequest):
    """
    Send a query to the chatbot
    
    Args:
        request: QueryRequest with query text and optional session_id
    
    Returns:
        QueryResponse with answer and metadata
    """
    try:
        logger.info(f"Query: {request.query[:100]}")
        
        start_time = time.time()
        result = chatbot.query(request.query)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Save to session if provided
        if request.session_id:
            try:
                storage.save_message(
                    session_id=request.session_id,
                    role="user",
                    content=request.query
                )
                storage.save_message(
                    session_id=request.session_id,
                    role="assistant",
                    content=result['answer'],
                    sources=result.get('sources', [])
                )
            except Exception as storage_error:
                logger.warning(f"Storage error: {storage_error}")
        
        return QueryResponse(
            answer=result['answer'],
            sources=result.get('sources', []),
            type=result.get('type', 'unknown'),
            latency_ms=elapsed_ms
        )
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/new")
async def new_session():
    """Start a new conversation session"""
    session_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return {
        "session_id": session_id,
        "status": "created",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/session/list", response_model=List[SessionInfo])
async def list_sessions():
    """List all saved conversation sessions"""
    try:
        sessions = storage.list_sessions()
        return [
            SessionInfo(
                session_id=s.get('session_id', 'unknown'),
                messages=s.get('messages', 0),
                created=s.get('created', datetime.now().isoformat())
            )
            for s in sessions
        ]
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return []

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get a specific conversation session"""
    try:
        data = storage.load(session_id)
        if not data:
            raise HTTPException(status_code=404, detail="Session not found")
        return data
    except Exception as e:
        logger.error(f"Error loading session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a conversation session"""
    try:
        if storage.delete(session_id):
            return {"status": "deleted", "session_id": session_id}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics", response_model=StatisticsResponse)
async def get_statistics():
    """Get statistics about all conversations"""
    try:
        stats = storage.get_stats()
        return StatisticsResponse(
            total_sessions=stats.get('total_sessions', 0),
            total_messages=stats.get('total_messages', 0),
            avg_messages_per_session=stats.get('avg_messages_per_session', 0.0)
        )
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return StatisticsResponse(
            total_sessions=0,
            total_messages=0,
            avg_messages_per_session=0.0
        )

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "kb_documents": chatbot.rag.collection.count(),
        "services": {
            "chatbot": "ok",
            "storage": "ok",
            "api": "ok"
        }
    }

# Run server
if __name__ == "__main__":
    print("🚀 Starting WikiChatbot API...")
    print("📚 Documentation: http://127.0.0.1:8000/docs")
    print("🔍 Health check: http://127.0.0.1:8000/health")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )