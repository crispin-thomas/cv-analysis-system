from fastapi import APIRouter
from app.schemas.schema import QueryRequest, QueryResponse

from app.core.config import Settings
from app.services.llm_service import LLMService
from app.services.query_engine import QueryEngine
from app.services.cv_database import CVDatabase

settings = Settings()
llm_service = LLMService(settings)
cv_database = CVDatabase(settings.STORAGE_DIR)
query_engine = QueryEngine(llm_service, cv_database)

router = APIRouter(prefix="", tags=["Query"])

@router.post("/query/", response_model=QueryResponse)
async def query_cvs(request: QueryRequest):
    """Query the CV database with natural language"""
    response = query_engine.process_query(
        request.query, user_id=request.user_id, conversation_id=request.conversation_id
    )
    return {"response": response}