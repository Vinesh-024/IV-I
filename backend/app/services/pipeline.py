import os
import logging
from app.services.models import ModalityRouter, BiomedCLIPService, RadDINOService, YOLOv11SegService
from app.services.severity import estimate_severity
from app.services.specialist import get_specialist_recommendation

logger = logging.getLogger("app.pipeline")

# Initialize models globally (singleton style)
router_model = ModalityRouter()
biomed_clip = BiomedCLIPService()
rad_dino = RadDINOService()
yolo_seg = YOLOv11SegService()

async def run_pipeline(img_path: str) -> dict:
    logger.info(f"Running analysis pipeline on: {img_path}")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at: {img_path}")
        
    # Step 1: Detect Modality
    modality = router_model.predict(img_path)
    logger.info(f"Modality detected: {modality}")
    
    # Step 2: Classify (Body part / Scan type + Disease)
    if modality == "RGB":
        clf_result = biomed_clip.predict(img_path)
    else:
        clf_result = rad_dino.predict(img_path)
        
    body_part_scan_type = clf_result["body_part_scan_type"]
    disease = clf_result["disease"]
    confidence = clf_result["confidence"]
    logger.info(f"Classification: {body_part_scan_type} -> {disease} ({confidence:.2%})")
    
    # Step 3: Localize (Bounding box + Segmentation Mask overlay)
    localization = yolo_seg.segment(img_path, disease)
    bbox = localization["bbox"]
    mask_filename = localization["mask_filename"]
    
    # Step 4: Estimate Severity
    severity = estimate_severity(
        modality=modality,
        body_part_scan_type=body_part_scan_type,
        disease=disease,
        bbox=bbox,
        affected_pixel_count=localization.get("affected_pixel_count", 0),
        total_pixel_count=localization.get("total_pixel_count", 1)
    )
    logger.info(f"Estimated Severity: {severity}")
    
    # Step 5: Specialist recommendation
    specialist = get_specialist_recommendation(body_part_scan_type, disease)
    logger.info(f"Specialist recommended: {specialist}")
    
    # Compile a quick, clear diagnostic report text summary (baseline before LLM expansion)
    report_text = (
        f"CLINICAL FINDINGS:\n"
        f"Visual inspection of the uploaded {body_part_scan_type} image "
        f"revealed pathological characteristics consistent with {disease}.\n\n"
        f"ANALYSIS DETAILS:\n"
        f"- Modality: {modality}\n"
        f"- Detected Condition: {disease}\n"
        f"- Severity Classification: {severity.upper()}\n"
        f"- Prediction Confidence: {confidence:.1%}\n\n"
        f"RECOMMENDED CONSULTATION:\n"
        f"The patient should be scheduled for a follow-up assessment with a {specialist} "
        f"for standard verification and management planning."
    )
    
    return {
        "modality": modality,
        "body_part_scan_type": body_part_scan_type,
        "disease": disease,
        "confidence": confidence,
        "bbox": bbox,
        "mask_filename": mask_filename,
        "severity": severity,
        "specialist": specialist,
        "report_text": report_text
    }
