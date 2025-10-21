from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging

# Imports locaux - vérifie qu'ils existent
try:
    from chatbot import WikiChatbot
    from chat_storage import ChatStorage
    CHATBOT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import chatbot modules: {e}")
    print("    API will run in limited mode")
    CHATBOT_AVAILABLE = False

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

# Initialize services (only if imports worked)
if CHATBOT_AVAILABLE:
    try:
        chatbot = WikiChatbot()
        storage = ChatStorage()
        logger.info("✅ Chatbot and storage initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize chatbot: {e}")
        CHATBOT_AVAILABLE = False

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    success: bool

# API Endpoints

@app.get("/")
async def root():
    """Health check - basic"""
    return {
        "status": "ok",
        "service": "Wiki Chatbot API",
        "version": "1.0.0",
        "chatbot_available": CHATBOT_AVAILABLE
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if CHATBOT_AVAILABLE else "degraded",
        "chatbot": "available" if CHATBOT_AVAILABLE else "unavailable",
        "api": "ok"
    }

@app.post("/query", response_model=QueryResponse)
async def query_chatbot(request: QueryRequest):
    """
    Send a query to the chatbot
    """
    if not CHATBOT_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Chatbot service is not available. Please check server logs."
        )
    
    try:
        logger.info(f"Query: {request.query[:100]}")
        
        result = chatbot.query(request.query)
        
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources'],
            success=result['success']
        )
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/new")
async def new_session():
    """Start a new conversation session"""
    if not CHATBOT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Chatbot not available")
    
    global chatbot
    chatbot = WikiChatbot()
    return {
        "session_id": chatbot.session_id,
        "status": "created"
    }

@app.get("/session/list")
async def list_sessions():
    """List all saved conversation sessions"""
    if not CHATBOT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Storage not available")
    
    try:
        sessions = storage.list_sessions()
        return sessions
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    """Get statistics about all conversations"""
    if not CHATBOT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Storage not available")
    
    try:
        stats = storage.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 Starting Wiki Chatbot API")
    print("="*70)
    print(f"Chatbot available: {CHATBOT_AVAILABLE}")
    print("="*70 + "\n")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )