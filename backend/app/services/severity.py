import logging

logger = logging.getLogger("app.severity")

def estimate_severity(
    modality: str,
    body_part_scan_type: str,
    disease: str,
    bbox: list,
    affected_pixel_count: int,
    total_pixel_count: int
) -> str:
    """
    Computes disease-aware severity classification (Mild, Moderate, Severe)
    based on bounding box size, pixel mask ratios, and disease priors.
    """
    # Calculate box ratio to image size
    box_ratio = 0.0
    if bbox and len(bbox) == 4:
        xmin, ymin, xmax, ymax = bbox
        box_area = (xmax - xmin) * (ymax - ymin)
        box_ratio = box_area / total_pixel_count
        
    mask_ratio = affected_pixel_count / total_pixel_count
    
    # Heuristic metrics (use whichever is higher/more representative)
    involvement_ratio = max(box_ratio, mask_ratio)
    
    disease_lower = disease.lower()
    
    # 1. Skin Disease
    if body_part_scan_type.lower() == "skin":
        if "melanoma" in disease_lower:
            # Melanoma is a high-risk malignancy; defaults to Severe/Moderate
            if involvement_ratio > 0.35:
                return "Severe"
            return "Moderate"
        elif "carcinoma" in disease_lower:
            if involvement_ratio > 0.40:
                return "Severe"
            return "Moderate"
        else: # Eczema, Psoriasis
            if involvement_ratio < 0.25:
                return "Mild"
            elif involvement_ratio < 0.50:
                return "Moderate"
            return "Severe"
            
    # 2. Burns
    elif "burn" in disease_lower:
        if "third-degree" in disease_lower or involvement_ratio > 0.45:
            return "Severe"
        elif "second-degree" in disease_lower or involvement_ratio > 0.20:
            return "Moderate"
        return "Mild"
        
    # 3. Wounds & Ulcers
    elif "ulcer" in disease_lower or "wound" in disease_lower or "incision" in disease_lower:
        if involvement_ratio > 0.40:
            return "Severe"
        elif involvement_ratio > 0.20:
            return "Moderate"
        return "Mild"
        
    # 4. Ophthalmology (Eye)
    elif body_part_scan_type.lower() == "eye":
        if "cataract" in disease_lower:
            return "Moderate"
        if involvement_ratio > 0.30:
            return "Moderate"
        return "Mild"
        
    # 5. Chest Radiology (Pneumonia, Effusion)
    elif "chest" in body_part_scan_type.lower() or "lung" in disease_lower:
        # Opacity and infiltration ratio
        if involvement_ratio > 0.45:
            return "Severe"
        elif involvement_ratio > 0.25:
            return "Moderate"
        return "Mild"
        
    # 6. Brain scans
    elif "brain" in body_part_scan_type.lower() or "tumor" in disease_lower or "hemorrhage" in disease_lower:
        if "tumor" in disease_lower or "hemorrhage" in disease_lower:
            if involvement_ratio > 0.30:
                return "Severe"
            return "Moderate"
        return "Mild"
        
    # 7. Bone & Joint fractures
    elif "fracture" in disease_lower:
        if involvement_ratio > 0.35:
            return "Severe"
        return "Moderate"
        
    # Default Heuristics for other categories
    if involvement_ratio < 0.20:
        return "Mild"
    elif involvement_ratio < 0.45:
        return "Moderate"
    return "Severe"
