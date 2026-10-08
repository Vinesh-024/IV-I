from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.database import get_report_by_id, save_report
from app.services.rag import get_rag_advice

router = APIRouter()

class ChatMessage(BaseModel):
    role: str # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    report_id: str
    query: str
    history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    answer: str
    suggestions: List[str]

@router.post("/query", response_model=ChatResponse)
async def query_rag_advisor(request: ChatRequest):
    report = await get_report_by_id(request.report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Corresponding patient case report not found")
    
    try:
        # Run vector retrieval + LLM execution to get the answer
        answer, suggestions = await get_rag_advice(report, request.query, request.history)
        
        # If this is the initial or primary question, let's save the generated suggestions to the report
        if not report.get("rag_suggestions") or len(report.get("rag_suggestions")) == 0:
            report["rag_suggestions"] = suggestions
            await save_report(report)
            
        return ChatResponse(answer=answer, suggestions=suggestions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG advice query execution failed: {e}")
