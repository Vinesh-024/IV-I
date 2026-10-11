import os
import logging
from app.services.models import ModalityRouter, BiomedCLIPService, RadDINOService, YOLOv11SegService
from app.services.severity import estimate_severity
from app.services.specialist import get_specialist_recommendation
from app.services.explainability import generate_biomedclip_gradcam, generate_dinov2_attention_heatmap
from app.config import settings

logger = logging.getLogger("app.pipeline")

# Initialize models globally (singleton style)
router_model = ModalityRouter()
biomed_clip = BiomedCLIPService()
rad_dino = RadDINOService()
yolo_seg = YOLOv11SegService()


async def _generate_llm_report(modality, body_part_scan_type, disease, severity, confidence,
                                 patient_age, patient_sex, symptoms) -> str:
    """
    Generate a narrative clinical findings paragraph using the configured LLM provider.
    Falls back to a structured template if no LLM is available.
    """
    import requests

    prompt = (
        f"You are a medical AI assistant generating a structured clinical findings summary for a report.\n"
        f"Write a concise, professional 3–4 sentence clinical findings paragraph for a patient report.\n"
        f"Do NOT use bullet points. Write in clinical prose. Do NOT give prescriptions.\n\n"
        f"Patient Details:\n"
        f"- Age: {patient_age} | Sex: {patient_sex}\n"
        f"- Chief Symptoms: {symptoms}\n"
        f"- Image Modality: {modality}\n"
        f"- Imaging Region/Type: {body_part_scan_type}\n"
        f"- AI-Detected Pathology: {disease}\n"
        f"- Confidence Score: {confidence:.1%}\n"
        f"- Severity Classification: {severity.upper()}\n\n"
        f"Write the clinical findings paragraph now:"
    )

    # 1. Try Gemini
    if settings.API_PROVIDER == "gemini" and settings.GEMINI_API_KEY:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={settings.GEMINI_API_KEY}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            res = requests.post(url, json=payload, timeout=12)
            if res.status_code == 200:
                text = res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                logger.info("LLM report narrative generated via Gemini.")
                return text
        except Exception as e:
            logger.warning(f"Gemini report generation failed: {e}")

    # 2. Try OpenAI
    if settings.API_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
            payload = {
                "model": "gpt-4-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300
            }
            res = requests.post(url, json=payload, headers=headers, timeout=12)
            if res.status_code == 200:
                text = res.json()["choices"][0]["message"]["content"].strip()
                logger.info("LLM report narrative generated via OpenAI.")
                return text
        except Exception as e:
            logger.warning(f"OpenAI report generation failed: {e}")

    # 3. Try local Ollama
    try:
        payload = {
            "model": "qwen2.5:1.5b",
            "prompt": prompt,
            "stream": False
        }
        res = requests.post(settings.LOCAL_LLM_URL, json=payload, timeout=8)
        if res.status_code == 200:
            text = res.json().get("response", "").strip()
            if text:
                logger.info("LLM report narrative generated via local Ollama.")
                return text
    except Exception as e:
        logger.info(f"Ollama offline or unavailable: {e}")

    # 4. Structured template fallback
    logger.info("LLM unavailable — using structured template report.")
    return (
        f"CLINICAL FINDINGS:\n"
        f"Visual inspection of the uploaded {body_part_scan_type} image revealed "
        f"pathological characteristics consistent with {disease}. "
        f"The AI classification pipeline assigned a confidence score of {confidence:.1%} "
        f"for this finding.\n\n"
        f"ANALYSIS DETAILS:\n"
        f"- Modality: {modality}\n"
        f"- Detected Condition: {disease}\n"
        f"- Severity Classification: {severity.upper()}\n"
        f"- Prediction Confidence: {confidence:.1%}\n\n"
        f"RECOMMENDED CONSULTATION:\n"
        f"The patient should be scheduled for a follow-up assessment with the recommended specialist "
        f"for standard verification and management planning."
    )


async def run_pipeline(img_path: str, patient_age: int = 0, patient_sex: str = "", symptoms: str = "") -> dict:
    logger.info(f"Running Phase 2 analysis pipeline on: {img_path}")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at: {img_path}")

    uploads_dir = os.path.dirname(img_path)

    # Step 1: Detect Modality
    modality = router_model.predict(img_path)
    logger.info(f"Modality detected: {modality}")

    # Step 2: Classify disease
    if modality == "RGB":
        clf_result = biomed_clip.predict(img_path)
    else:
        clf_result = rad_dino.predict(img_path)

    body_part_scan_type = clf_result["body_part_scan_type"]
    disease = clf_result["disease"]
    confidence = clf_result["confidence"]
    # Capture predicted class index for explainability (if available)
    pred_idx = clf_result.get("pred_idx", 0)
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

    # Step 6 (Phase 2 NEW): Generate LLM Clinical Report Narrative
    report_text = await _generate_llm_report(
        modality=modality,
        body_part_scan_type=body_part_scan_type,
        disease=disease,
        severity=severity,
        confidence=confidence,
        patient_age=patient_age,
        patient_sex=patient_sex,
        symptoms=symptoms
    )

    # Step 7 (Phase 2 NEW): Generate Explainability Heatmap
    gradcam_filename = ""
    try:
        if modality == "RGB":
            gradcam_filename = generate_biomedclip_gradcam(
                img_path=img_path,
                biomed_clip_service=biomed_clip,
                pred_idx=pred_idx,
                uploads_dir=uploads_dir
            )
        else:
            gradcam_filename = generate_dinov2_attention_heatmap(
                img_path=img_path,
                rad_dino_service=rad_dino,
                uploads_dir=uploads_dir
            )
        logger.info(f"Explainability heatmap generated: {gradcam_filename}")
    except Exception as e:
        logger.error(f"Explainability step failed (non-critical): {e}")

    return {
        "modality": modality,
        "body_part_scan_type": body_part_scan_type,
        "disease": disease,
        "confidence": confidence,
        "bbox": bbox,
        "mask_filename": mask_filename,
        "gradcam_filename": gradcam_filename,
        "severity": severity,
        "specialist": specialist,
        "report_text": report_text
    }
