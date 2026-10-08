import os
import json
import logging
import requests
import numpy as np
from app.config import settings

logger = logging.getLogger("app.rag")

# Vector Database Simulation (Numpy Cosine Similarity over local embeddings)
EMBEDDING_MODEL = None
GUIDELINES = []
GUIDELINES_EMBEDDINGS = None
GUIDELINES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "medical_guidelines.json")

def init_rag_system():
    global EMBEDDING_MODEL, GUIDELINES, GUIDELINES_EMBEDDINGS
    
    # 1. Load clinical guidelines JSON data
    os.makedirs(os.path.dirname(GUIDELINES_PATH), exist_ok=True)
    if not os.path.exists(GUIDELINES_PATH):
        # Seed default clinical data if not present
        default_guidelines = [
            {
                "condition": "Melanoma / Skin Cancer",
                "text": "Clinical Guidelines for Melanoma: Immediate dermatologist referral required. Check lesion borders (Asymmetric, Border irregular, Color variegated, Diameter >6mm - ABCD criteria). Biopsy is gold standard. Management: Surgical excision, staging sentinel node biopsy. Keep lesion dry, avoid sun exposure, apply SPF 50+."
            },
            {
                "condition": "Eczema / Dermatitis",
                "text": "Clinical Guidelines for Atopic Dermatitis/Eczema: Maintain skin hydration. Use emollients daily. Topical corticosteroids (e.g. hydrocortisone) for flare-ups. Avoid harsh soaps, fragrances, and hot showers. Manage itch with oral antihistamines."
            },
            {
                "condition": "Pneumonia",
                "text": "Clinical Guidelines for Pneumonia: Assess severity using CURB-65 criteria (Confusion, Urea, Respiratory rate, Blood pressure, Age >= 65). Administer empirical antibiotics (e.g. Amoxicillin or Azithromycin). Rest, hydration, and bronchodilators for dyspnea. Monitor oxygen saturation. Repeat chest X-ray in 6 weeks."
            },
            {
                "condition": "Second-Degree Burn",
                "text": "Clinical Guidelines for Moderate Burns (Second-Degree): Clean with mild soap and cool water. Apply silver sulfadiazine or antibiotic ointment. Cover with sterile non-adherent dressing. Daily dressing changes. Do not pop blisters. Manage pain with NSAIDs. Specialist consult for joints or face."
            },
            {
                "condition": "Finger Fracture / Bone Fracture",
                "text": "Clinical Guidelines for Fractures: Immobilize joint above and below the fracture (splint/cast). RICE protocol (Rest, Ice, Compression, Elevation). Refer to orthopedics for reduction and alignment. Monitor distal pulses and sensation (capillary refill check)."
            },
            {
                "condition": "Conjunctivitis",
                "text": "Clinical Guidelines for Conjunctivitis (Pink Eye): Viral cases are self-limiting; use artificial tears, cold compresses. Bacterial cases require antibiotic drops (e.g., Polymyxin/Trimethoprim). Avoid contact lens wear. Hand hygiene to prevent transmission."
            },
            {
                "condition": "Paronychia / Nail Infection",
                "text": "Clinical Guidelines for Paronychia: Warm water soaks 3-4 times daily. If early/mild, topical mupirocin ointment is indicated. For fluctuant abscess, perform simple incision and drainage. Oral antibiotics (Cephalexin) if cellulitis is present."
            },
            {
                "condition": "Pressure Ulcer / Wound",
                "text": "Clinical Guidelines for Pressure Ulcers: Offload pressure (repositioning every 2 hours). Keep wound bed moist and clean. Debride necrotic tissue if needed. Apply hydrocolloid or alginate dressings. Ensure high-protein nutrition."
            }
        ]
        with open(GUIDELINES_PATH, "w", encoding="utf-8") as f:
            json.dump(default_guidelines, f, indent=2)
            
    try:
        with open(GUIDELINES_PATH, "r", encoding="utf-8") as f:
            GUIDELINES = json.load(f)
    except Exception as e:
        logger.error(f"Error loading guidelines database file: {e}")
        GUIDELINES = []

    # 2. Try loading SentenceTransformer for local embeddings
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers/all-MiniLM-L6-v2 model locally...")
        EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Pre-compute guideline embeddings
        texts = [g["text"] for g in GUIDELINES]
        if texts:
            GUIDELINES_EMBEDDINGS = EMBEDDING_MODEL.encode(texts, convert_to_numpy=True)
            logger.info(f"Loaded guidelines database. Computed {len(GUIDELINES_EMBEDDINGS)} embeddings.")
    except Exception as e:
        logger.warning(f"Could not load local embedding model: {e}. RAG will use keyword lookup.")
        EMBEDDING_MODEL = None


def retrieve_relevant_guidelines(query: str, top_k: int = 1) -> str:
    """
    Retrieves the top-k clinical guidelines matching the query text.
    """
    if not GUIDELINES:
        return ""
        
    # If embedding model is active, use Cosine Similarity
    if EMBEDDING_MODEL is not None and GUIDELINES_EMBEDDINGS is not None:
        try:
            query_emb = EMBEDDING_MODEL.encode([query], convert_to_numpy=True)[0]
            # Compute cosine similarities
            norms = np.linalg.norm(GUIDELINES_EMBEDDINGS, axis=1) * np.linalg.norm(query_emb)
            norms[norms == 0] = 1e-10
            similarities = np.dot(GUIDELINES_EMBEDDINGS, query_emb) / norms
            best_idx = np.argmax(similarities)
            logger.info(f"RAG Match: '{GUIDELINES[best_idx]['condition']}' (Score: {similarities[best_idx]:.3f})")
            return GUIDELINES[best_idx]["text"]
        except Exception as e:
            logger.error(f"Cosine similarity retrieval failed: {e}. Falling back to keyword matching.")

    # Keyword backup matcher
    query_lower = query.lower()
    for g in GUIDELINES:
        cond_words = g["condition"].lower().split()
        if any(w in query_lower for w in cond_words if len(w) > 3):
            return g["text"]
            
    # Default to first guideline if no matches
    return GUIDELINES[0]["text"] if GUIDELINES else ""


async def get_rag_advice(report: dict, query: str, history: list = None) -> tuple:
    """
    Core RAG function. Retrieves relevant medical evidence guidelines,
    combines it with the patient's case details and user query, and queries the LLM.
    Returns: (str: LLM response, list: parsed checklist suggestions).
    """
    # 1. Initialize DB/embeddings if not done yet
    if not GUIDELINES:
        init_rag_system()
        
    # 2. Retrieve clinical guideline evidence
    evidence_query = f"{report.get('disease')} {report.get('body_part_scan_type')} {query}"
    clinical_guideline = retrieve_relevant_guidelines(evidence_query)
    
    # 3. Construct LLM Prompt
    system_prompt = (
        "You are an expert AI clinical medical assistant. Your role is to provide patient educational guidance "
        "and suggest general care checklists based on diagnostic reports and verified clinical guidelines.\n"
        "RULES:\n"
        "1. Do NOT write definitive medical prescriptions or treatment plans.\n"
        "2. Keep advice educational and focused on lifestyle, symptoms to watch out for, and specialist consultation prep.\n"
        "3. End your response with a 3-5 bullet checklist of specific suggestions/steps for the patient.\n"
        "Format the checklist clearly with brackets at the start of each step like: '[ ] Step description'\n"
    )
    
    context = (
        f"PATIENT REPORT DETAILS:\n"
        f"- Patient Name: {report.get('patient_name')}\n"
        f"- Age: {report.get('patient_age')} | Sex: {report.get('patient_sex')}\n"
        f"- Reported Symptoms: {report.get('symptoms')}\n"
        f"- Primary AI Finding: {report.get('disease')} ({report.get('body_part_scan_type')})\n"
        f"- Estimated Severity: {report.get('severity')}\n"
        f"- Recommended Specialist: {report.get('specialist')}\n\n"
        f"VERIFIED CLINICAL EVIDENCE:\n"
        f"{clinical_guideline}\n\n"
        f"USER PATIENT QUERY:\n"
        f"{query}\n"
    )
    
    prompt = f"{system_prompt}\n{context}"
    
    answer = ""
    # 4. Fetch Response from configured Provider
    if settings.API_PROVIDER == "gemini" and settings.GEMINI_API_KEY:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={settings.GEMINI_API_KEY}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            res = requests.post(url, json=payload, timeout=10)
            if res.status_code == 200:
                answer = res.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.error(f"Gemini API query error: {e}")
            
    elif settings.API_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
            payload = {
                "model": "gpt-4-turbo",
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": context}]
            }
            res = requests.post(url, json=payload, headers=headers, timeout=10)
            if res.status_code == 200:
                answer = res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenAI API query error: {e}")
            
    elif settings.API_PROVIDER == "local" or (not answer):
        # Fallback to local Ollama instance or built-in template compiler if Ollama offline
        try:
            payload = {
                "model": "qwen2.5:1.5b" if "qwen" in settings.LOCAL_LLM_URL else "llama3.1",
                "prompt": prompt,
                "stream": False
            }
            res = requests.post(settings.LOCAL_LLM_URL, json=payload, timeout=5)
            if res.status_code == 200:
                answer = res.json().get("response", "")
        except Exception as e:
            logger.warning(f"Ollama local LLM query offline or timed out: {e}. Compiling clinical template advice.")
            
    # Heuristic Rule-Based Template Advisor (failsafe fallback)
    if not answer:
        answer = (
            f"Thank you for your inquiry regarding the AI clinical findings. Based on the diagnostic summary, "
            f"your case indicates potential <b>{report.get('disease')}</b> with a classification of <b>{report.get('severity')}</b>. "
            f"We strongly recommend scheduling an appointment with an experienced <b>{report.get('specialist')}</b> for a formal review.\n\n"
            f"In the meantime, here are general educational points from our guideline database regarding this condition:\n"
            f"• <i>General Management:</i> Avoid self-treatment or picking at the affected site. Keep the area clean and monitor for progressive redness, swelling, or pain.\n"
            f"• <i>Specialist Prep:</i> Compile a list of when the symptoms first appeared, visual changes over time, and any previous prescription medications used.\n\n"
            f"Please follow the customized checklist below to prepare for your clinical consultation:\n"
            f"[ ] Prepare a timeline of symptoms and visual changes to show the specialist.\n"
            f"[ ] Gather documentation on your existing medical history and daily medications.\n"
            f"[ ] Avoid applying home remedies or strong chemical products to the area.\n"
            f"[ ] Seek immediate emergency care if you experience systemic symptoms like fever or severe escalating pain."
        )

    # 5. Extract checklist items formatted like '[ ] Step'
    suggestions = []
    for line in answer.split("\n"):
        if "[ ]" in line:
            clean_sug = line.replace("[ ]", "").strip("-* •").strip()
            if clean_sug:
                suggestions.append(clean_sug)
                
    # If no structured checklists were extracted, provide default ones
    if not suggestions:
        suggestions = [
            f"Schedule a follow-up assessment with a {report.get('specialist')}.",
            "Create a timeline detailing symptom progression and trigger factors.",
            "Avoid altering the lesion or applying unprescribed medical ointments.",
            "Prepare a list of current prescriptions and drug allergies for your doctor."
        ]
        
    return answer, suggestions
