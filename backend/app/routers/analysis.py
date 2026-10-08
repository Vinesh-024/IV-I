import os
import uuid
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import shutil

from app.config import settings
from app.database import save_report
from app.services.pipeline import run_pipeline

router = APIRouter()
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "uploads")

@router.post("/upload")
async def upload_and_analyze(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    patient_name: str = Form(...),
    patient_age: int = Form(...),
    patient_sex: str = Form(...),
    symptoms: str = Form(...),
    medical_history: Optional[str] = Form("")
):
    # Ensure uploads directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Generate unique filename to avoid overwrites
    file_ext = os.path.splitext(image.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save the file locally
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded image: {e}")
    
    # Run the image and metadata through the AI routing/analysis pipeline
    try:
        pipeline_results = await run_pipeline(file_path)
    except Exception as e:
        # Cleanup uploaded image on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"AI pipeline analysis failed: {e}")
    
    # Combine patient details with pipeline analysis outputs
    report_id = str(uuid.uuid4())
    report_data = {
        "id": report_id,
        "patient_name": patient_name,
        "patient_age": patient_age,
        "patient_sex": patient_sex,
        "symptoms": symptoms,
        "medical_history": medical_history,
        "image_filename": unique_filename,
        "modality": pipeline_results.get("modality"),
        "body_part_scan_type": pipeline_results.get("body_part_scan_type"),
        "disease": pipeline_results.get("disease"),
        "confidence": pipeline_results.get("confidence"),
        "severity": pipeline_results.get("severity"),
        "specialist": pipeline_results.get("specialist"),
        "report_text": pipeline_results.get("report_text"),
        # To be filled by RAG step later, or pre-filled with baseline advice
        "rag_suggestions": [],
        "created_at": datetime.now().isoformat(),
        # Store detection coordinates and mask path for front-end rendering
        "bbox": pipeline_results.get("bbox", []),
        "mask_filename": pipeline_results.get("mask_filename", "")
    }
    
    # Save report to DB
    await save_report(report_data)
    
    return report_data
