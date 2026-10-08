def get_specialist_recommendation(body_part_scan_type: str, disease: str) -> str:
    """
    Standard lookup mapping disease categories or modalities to medical specialists.
    """
    category = body_part_scan_type.lower()
    dis = disease.lower()
    
    if "skin" in category or "melanoma" in dis or "carcinoma" in dis or "eczema" in dis or "psoriasis" in dis:
        return "Dermatologist"
        
    elif "burn" in dis:
        if "third-degree" in dis:
            return "Burn Surgeon / Plastic Surgeon"
        return "General Surgeon / Dermatologist"
        
    elif "ulcer" in dis or "wound" in dis:
        return "Wound Care Specialist / General Surgeon"
        
    elif "eye" in category or "conjunctivitis" in dis or "stye" in dis or "blepharitis" in dis or "cataract" in dis:
        return "Ophthalmologist"
        
    elif "ear" in category or "otitis" in dis:
        return "Otolaryngologist (ENT Specialist)"
        
    elif "lips" in category or "tongue" in category or "candidiasis" in dis or "cheilitis" in dis:
        # Oral/dental/dermatological crossover
        if "candidiasis" in dis:
            return "Oral Medicine Specialist / General Physician"
        return "Dentist / Dermatologist"
        
    elif "chest" in category or "lung" in dis or "pneumonia" in dis or "atelectasis" in dis or "effusion" in dis:
        return "Pulmonologist"
        
    elif "brain" in category or "tumor" in dis or "hemorrhage" in dis or "stroke" in dis or "sclerosis" in dis:
        if "tumor" in dis or "hemorrhage" in dis:
            return "Neurosurgeon / Oncologist"
        return "Neurologist"
        
    elif "spine" in category or "herniated" in dis or "stenosis" in dis or "spondylolisthesis" in dis:
        return "Orthopedic Spine Specialist / Neurosurgeon"
        
    elif "fracture" in dis:
        return "Orthopedic Surgeon"
        
    # General default routing
    if "x-ray" in category or "ct" in category or "mri" in category:
        return "Radiologist / General Physician"
    return "General Physician"
