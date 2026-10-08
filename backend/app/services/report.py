import os
import logging
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io

logger = logging.getLogger("app.report")

def generate_pdf_report(report_data: dict) -> bytes:
    """
    Generates a structured medical report PDF using ReportLab.
    Returns the PDF as raw bytes.
    """
    buffer = io.BytesIO()
    
    # Page setup - 0.75 margin
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette - Premium Teal and Navy Blue
    COLOR_PRIMARY = colors.HexColor("#0f766e")   # Teal 700
    COLOR_SECONDARY = colors.HexColor("#1e293b") # Slate 800
    COLOR_BG_LIGHT = colors.HexColor("#f8fafc")  # Slate 50
    COLOR_HIGHLIGHT = colors.HexColor("#06b6d4") # Cyan 500
    COLOR_BORDER = colors.HexColor("#cbd5e1")    # Slate 300
    COLOR_DANGER = colors.HexColor("#ef4444")    # Red 500
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=COLOR_PRIMARY,
        spaceAfter=15
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=COLOR_PRIMARY,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=COLOR_SECONDARY
    )
    
    body_bold_style = ParagraphStyle(
        'ReportBodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    meta_header_style = ParagraphStyle(
        'MetaHeader',
        parent=body_style,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor("#475569") # Slate 600
    )
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=body_style,
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748b") # Slate 500
    )
    
    story = []
    
    # 1. Header (Hospital/Institution Header & Title)
    header_data = [
        [
            Paragraph("CLINICAL DECISION SUPPORT REPORT", title_style),
            Paragraph("<b>Case ID:</b> " + report_data.get("id", "N/A")[:8], ParagraphStyle('HeaderID', parent=body_style, alignment=2))
        ]
    ]
    header_table = Table(header_data, colWidths=[4.5*inch, 2.5*inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(header_table)
    
    # Divider line
    story.append(Table([[""]], colWidths=[7.0*inch], rowHeights=[2], style=[
        ('BACKGROUND', (0,0), (-1,-1), COLOR_PRIMARY),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(Spacer(1, 15))
    
    # 2. Patient Demographics & Intake Details
    story.append(Paragraph("PATIENT DEMOGRAPHICS", section_heading))
    patient_data = [
        [
            Paragraph("Patient Name:", meta_header_style),
            Paragraph(report_data.get("patient_name", "N/A"), body_style),
            Paragraph("Date Created:", meta_header_style),
            Paragraph(report_data.get("created_at", "N/A")[:10], body_style)
        ],
        [
            Paragraph("Age:", meta_header_style),
            Paragraph(str(report_data.get("patient_age", "N/A")), body_style),
            Paragraph("Sex:", meta_header_style),
            Paragraph(report_data.get("patient_sex", "N/A"), body_style)
        ],
        [
            Paragraph("Chief Symptoms:", meta_header_style),
            Paragraph(report_data.get("symptoms", "None reported"), body_style),
            Paragraph("Medical History:", meta_header_style),
            Paragraph(report_data.get("medical_history", "None reported") or "None reported", body_style)
        ]
    ]
    patient_table = Table(patient_data, colWidths=[1.3*inch, 2.2*inch, 1.3*inch, 2.2*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COLOR_BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 15))
    
    # 3. AI Pipeline Diagnostic Findings
    story.append(Paragraph("ARTIFICIAL INTELLIGENCE DIAGNOSTIC PIPELINE RESULTS", section_heading))
    
    severity_val = report_data.get("severity", "N/A").upper()
    severity_color = COLOR_PRIMARY
    if severity_val == "SEVERE":
        severity_color = COLOR_DANGER
    elif severity_val == "MODERATE":
        severity_color = colors.HexColor("#f59e0b") # Amber 500
        
    findings_data = [
        [
            Paragraph("Image Modality:", meta_header_style),
            Paragraph(report_data.get("modality", "N/A"), body_style),
            Paragraph("Region / Scan:", meta_header_style),
            Paragraph(report_data.get("body_part_scan_type", "N/A"), body_style)
        ],
        [
            Paragraph("Detected Pathology:", meta_header_style),
            Paragraph(f"<b>{report_data.get('disease', 'N/A')}</b>", body_style),
            Paragraph("Confidence Score:", meta_header_style),
            Paragraph(f"{report_data.get('confidence', 0.0):.1%}", body_style)
        ],
        [
            Paragraph("Severity Assessment:", meta_header_style),
            Paragraph(f"<font color='{severity_color}'><b>{severity_val}</b></font>", body_style),
            Paragraph("Recommended Specialist:", meta_header_style),
            Paragraph(f"<b>{report_data.get('specialist', 'N/A')}</b>", body_style)
        ]
    ]
    findings_table = Table(findings_data, colWidths=[1.5*inch, 2.0*inch, 1.5*inch, 2.0*inch])
    findings_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 0.5, COLOR_PRIMARY),
        ('INNERGRID', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(findings_table)
    story.append(Spacer(1, 15))
    
    # 4. Detailed Summary Findings Text
    story.append(Paragraph("CLINICAL SUMMARY", section_heading))
    story.append(Paragraph(report_data.get("report_text", "No clinical summary available.").replace("\n", "<br/>"), body_style))
    story.append(Spacer(1, 15))
    
    # 5. RAG Recommendations (if present)
    suggestions = report_data.get("rag_suggestions", [])
    if suggestions:
        story.append(Paragraph("AI SUGGESTIONS & PREVENTATIVE MANAGEMENT", section_heading))
        bullet_story = []
        for sug in suggestions:
            bullet_story.append(Paragraph(f"• {sug}", body_style))
            bullet_story.append(Spacer(1, 4))
        story.extend(bullet_story)
        story.append(Spacer(1, 15))
        
    # 6. Legal Disclaimer & Institutional Footer
    story.append(Spacer(1, 10))
    story.append(Table([[""]], colWidths=[7.0*inch], rowHeights=[0.5], style=[
        ('BACKGROUND', (0,0), (-1,-1), COLOR_BORDER),
    ]))
    story.append(Spacer(1, 5))
    disclaimer_text = (
        "DISCLAIMER: This document is an automated clinical report generated by an Artificial Intelligence "
        "decision support pipeline. It is intended solely for educational assistance and pre-clinical guidance. "
        "The findings have not been verified by a board-certified clinician. This report does NOT constitute formal "
        "medical diagnosis, treatment, or advice. Patients must consult with the recommended specialist or a licensed "
        "medical practitioner to obtain formal clinical care and diagnostic validation."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    
    # Build Document
    doc.build(story)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
