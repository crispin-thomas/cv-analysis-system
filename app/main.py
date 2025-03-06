import os
import uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tempfile

from app.core.config import Settings
from app.services.document_processor import DocumentProcessor
from app.services.information_extractor import InformationExtractor
from app.services.llm_service import LLMService
from app.services.query_engine import QueryEngine
from app.services.cv_database import CVDatabase

app = FastAPI(title="CV Analysis System API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get settings
settings = Settings()

# Initialize services
llm_service = LLMService(settings)
document_processor = DocumentProcessor(settings)
information_extractor = InformationExtractor(llm_service)
cv_database = CVDatabase(settings.STORAGE_DIR)
query_engine = QueryEngine(llm_service, cv_database)


# Models
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


class QueryResponse(BaseModel):
    response: str


@app.post("/upload-cv/", status_code=201)
async def upload_cv(cv_file: UploadFile = File(...)):
    """Upload and process a CV file"""
    # Generate unique ID for this CV
    cv_id = str(uuid.uuid4())

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(dir=settings.TEMP_STORAGE_DIR,delete=False,suffix=f".{cv_file.filename.split('.').pop()}") as temp_file:
        temp_file_path = temp_file.name
        content = await cv_file.read()
        temp_file.write(content)

    try:
        # Process document
        text_content = document_processor.process_document(temp_file_path)

        # Extract information
        cv_data = information_extractor.extract_cv_information(text_content)

        # Store processed data
        cv_database.save_cv(cv_id, cv_data)

        return {"cv_id": cv_id, "filename": cv_file.filename, "status": "processed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CV: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/query/", response_model=QueryResponse)
async def query_cvs(request: QueryRequest):
    """Query the CV database with natural language"""
    response = query_engine.process_query(
        request.query, user_id=request.user_id, conversation_id=request.conversation_id
    )
    return {"response": response}


@app.get("/cvs/", status_code=200)
async def list_cvs():
    """List all processed CVs in the database"""
    cvs = cv_database.get_all_cvs()
    return {"cv_count": len(cvs), "cvs": cvs}


@app.get("/cvs/{cv_id}", status_code=200)
async def get_cv(cv_id: str):
    """Get a specific CV by ID"""
    cv_data = cv_database.get_cv(cv_id)
    if not cv_data:
        raise HTTPException(status_code=404, detail=f"CV with ID {cv_id} not found")
    return cv_data


@app.delete("/cvs/{cv_id}", status_code=200)
async def delete_cv(cv_id: str):
    """Delete a CV by ID"""
    success = cv_database.delete_cv(cv_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"CV with ID {cv_id} not found")
    return {"status": "deleted", "cv_id": cv_id}
