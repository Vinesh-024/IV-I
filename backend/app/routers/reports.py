from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from app.database import get_all_reports, get_report_by_id
from app.services.report import generate_pdf_report
import io

router = APIRouter()

@router.get("/")
async def list_reports():
    try:
        reports = await get_all_reports()
        return reports
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch reports: {e}")

@router.get("/{report_id}")
async def get_report(report_id: str):
    report = await get_report_by_id(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report

@router.get("/{report_id}/pdf")
async def download_pdf(report_id: str):
    report = await get_report_by_id(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    try:
        # Generate the PDF file as a byte stream
        pdf_bytes = generate_pdf_report(report)
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=report_{report_id}.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {e}")
