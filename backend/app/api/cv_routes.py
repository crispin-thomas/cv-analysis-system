import os
import uuid
import tempfile
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, HTTPException

from app.core.config import Settings
from app.services.llm_service import LLMService
from app.services.document_processor import DocumentProcessor
from app.services.information_extractor import InformationExtractor
from app.services.cv_database import CVDatabase

settings = Settings()
llm_service = LLMService(settings)
document_processor = DocumentProcessor(settings)
information_extractor = InformationExtractor(llm_service)
cv_database = CVDatabase(settings.STORAGE_DIR)

router = APIRouter(prefix="", tags=["CVs"])

@router.post("/upload/", status_code=201)
async def upload_cv(cv_file: UploadFile = File(...)):
    """Upload and process a CV file"""
    # Generate unique ID for this CV
    cv_id = str(uuid.uuid4())

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(dir=settings.TEMP_STORAGE_DIR,delete=False,suffix=f".{cv_file.filename.split('.').pop()}") as temp_file:
        temp_file_path = temp_file.name
        content = await cv_file.read()
        temp_file.write(content)
    
    file_info = {
        "filename": cv_file.filename,
        "file_size": len(content),  # File size in bytes
        "file_type": cv_file.content_type,
        "upload_date": datetime.utcnow().isoformat()  # UTC timestamp of upload
    }

    try:
        # Process document
        text_content = document_processor.process_document(temp_file_path)

        # Extract information
        cv_data = information_extractor.extract_cv_information(text_content)

        # Add file metadata to cv_data
        cv_data["meta"] = file_info
        
        # Store processed data
        cv_database.save_cv(cv_id, cv_data)

        return {"cv_id": cv_id, "filename": cv_file.filename, "status": "processed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CV: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.get("/cvs/", status_code=200)
async def list_cvs():
    """List all processed CVs in the database"""
    cvs = cv_database.get_all_cvs()
    return cvs

@router.get("/cvs/{cv_id}", status_code=200)
async def get_cv(cv_id: str):
    """Get a specific CV by ID"""
    cv_data = cv_database.get_cv(cv_id)
    if not cv_data:
        raise HTTPException(status_code=404, detail=f"CV with ID {cv_id} not found")
    return cv_data

@router.delete("/cvs/{cv_id}", status_code=200)
async def delete_cv(cv_id: str):
    """Delete a CV by ID"""
    success = cv_database.delete_cv(cv_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"CV with ID {cv_id} not found")
    return {"status": "deleted", "cv_id": cv_id}