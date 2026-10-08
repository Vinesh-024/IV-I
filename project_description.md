# Intelligent Medical Image Analysis with Automated Report Generation
### Complete Project Description — For Examiners & New Team Members

---

## 1. Project Overview

**Full Title:** *Intelligent Medical Image Analysis with Automated Report Generation using Vision-Language Foundation Models*

This is an end-to-end clinical AI decision-support system that accepts a medical image uploaded by a clinician or patient, automatically determines what type of image it is (a visible photo or a radiology scan), classifies the disease or pathology present, localizes the affected region with a bounding box and pixel-level segmentation mask, estimates severity, recommends the appropriate specialist, generates a structured PDF clinical report, and provides an interactive RAG (Retrieval-Augmented Generation) AI chat assistant for follow-up clinical questions — all running in real time through a modern web interface.

The project is built around the concept of **multi-modal medical image triage** — meaning the system can handle both **RGB visible-light photographs** (e.g. skin disease photos, eye photos, wound images) and **medical radiology scans** (X-ray, MRI, CT) from a single unified pipeline.

---

## 2. Base / Reference Paper

**Title:** *Vision-Language Models in Medical Applications: A Review*

The PDF (`Vision-Language_Models_in_Medical_Applications_A_Review.pdf`) is stored in the project root and served as the foundational academic reference. Key concepts adopted from the paper:

| Concept | How It Is Applied |
|---|---|
| Multi-modal Vision-Language Models | BiomedCLIP is used for RGB image classification using both zero-shot and supervised modes |
| Foundation Model Feature Extraction | DINOv2 features are extracted and passed to a trainable linear probe head |
| Zero-shot classification via text-image alignment | BiomedCLIP's CLIP backbone aligns image embeddings with clinical text prompts |
| Transfer learning / Linear Probe paradigm | Frozen backbone embeddings feed into a lightweight trainable classification head |
| RAG (Retrieval-Augmented Generation) for clinical advisory | Retrieved clinical guidelines + LLM generates patient-specific advisory |
| Localization / Segmentation in medical AI | YOLOv11 + SAM 2 hybrid pipeline for lesion detection and mask overlay |

---

## 3. System Architecture — Full Pipeline

When a user uploads an image, the system passes it through the following sequential AI pipeline:

```
                ┌─────────────────────────────────────┐
                │         USER: Upload Image           │
                │  + Patient Name, Age, Sex, Symptoms  │
                └────────────────┬────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
              STEP 1│    Modality Router     │  ← SimpleModalityCNN
                    │  RGB or Scan?          │     (Trained CNN)
                    └────────────┬───────────┘
                                 │
               ┌─────────────────┴──────────────────┐
               │ RGB                                 │ Scan
               ▼                                     ▼
  ┌──────────────────────────┐      ┌──────────────────────────────┐
  │ STEP 2a: BiomedCLIP      │      │ STEP 2b: RadDINO / DINOv2    │
  │ RGB Disease Classifier   │      │ Radiology Scan Classifier     │
  │ (Microsoft BiomedCLIP)   │      │ (Facebook DINOv2-ViTS14)      │
  └──────────────┬───────────┘      └──────────────┬───────────────┘
                 │                                   │
                 └─────────────┬─────────────────────┘
                               │
                               ▼
                  ┌────────────────────────┐
           STEP 3 │  YOLOv11 + SAM 2       │  ← Hybrid localization
                  │  Localization & Mask    │     bounding box + pixel mask
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
           STEP 4 │  Severity Estimator    │  ← Rule-based disease-aware
                  │  (Mild/Moderate/Severe) │     heuristic scoring
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
           STEP 5 │  Specialist Router     │  ← Dermatologist / Neurologist
                  │  Recommendation        │     / Pulmonologist etc.
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
           STEP 6 │  Report Generator      │  ← PDF via ReportLab
                  │  PDF Export            │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
           STEP 7 │  RAG Chat Assistant    │  ← SentenceTransformer +
                  │  (Clinical Q&A)        │     Gemini / OpenAI / Ollama
                  └────────────────────────┘
```

---

## 4. All Models — Detailed Descriptions

### 4.1 — `SimpleModalityCNN` (Modality Router)
**File:** `backend/app/services/models.py` → `ModalityRouter` class  
**Trained weights:** `backend/models/modality_model.pth` (8.1 MB)  
**Training script:** `backend/app/scripts/train_modality.py`

**What it does:**  
A lightweight custom CNN trained from scratch to classify any incoming image as either **Scan** (class 0 — grayscale radiology: X-ray, MRI, CT) or **RGB** (class 1 — visible-light color photographs: skin, wound, eye, nail, etc.).

**Architecture:**
```
Input: 128×128 RGB image

Feature Extractor:
  Conv2d(3→16, 3×3) → ReLU → MaxPool(2×2)   [128→64]
  Conv2d(16→32, 3×3) → ReLU → MaxPool(2×2)  [64→32]
  Conv2d(32→64, 3×3) → ReLU → MaxPool(2×2)  [32→16]

Classifier Head:
  Flatten → Linear(16384→128) → ReLU → Dropout(0.2) → Linear(128→2)

Output: Softmax → [P(Scan), P(RGB)]
```

**Training details:**
- Optimizer: Adam (lr=0.001)
- Epochs: 8
- Batch size: 16
- Loss: Cross-Entropy
- Train/Val split: 85%/15%

**Fallback:** If the CNN is not loaded, a color-channel standard deviation heuristic is used (grayscale scans have inter-channel std < 2.5).

---

### 4.2 — `BiomedCLIPService` (RGB Disease Classifier)
**File:** `backend/app/services/models.py` → `BiomedCLIPService` class  
**Foundation model:** `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (via HuggingFace Hub)  
**Trained head weights:** `backend/models/biomed_clip_head.pth` (263 KB)  
**Training script:** `backend/app/scripts/train_classifiers.py`

**What it does:**  
Classifies RGB visible-light medical images into **9 disease categories** using **Microsoft BiomedCLIP** — a Vision-Language foundation model pre-trained on 15 million biomedical image-text pairs from PubMed.

**Foundation Model Details:**
- Architecture: ViT-B/16 (Vision Transformer Base, 16×16 patch, 224×224 input)
- Text encoder: PubMedBERT (biomedical domain-specific BERT)
- Output embedding: 512-dimensional normalized image feature vector
- Pre-training data: 15M PubMed biomedical image-caption pairs

**Supervised Head (LinearProbeClassifier):**
```
Input: 512-dim BiomedCLIP image embedding (L2-normalized)
  Linear(512→128) → ReLU → Dropout(0.2) → Linear(128→9)
Output: Softmax → class probabilities over 9 RGB disease classes
```

**RGB Disease Classes (9 classes):**
| Class Folder | Disease |
|---|---|
| `Melanoma` | Melanoma (Skin Cancer) |
| `Basal_Cell_Carcinoma` | Basal Cell Carcinoma |
| `Eczema` | Atopic Dermatitis (Eczema) |
| `Psoriasis` | Psoriasis |
| `Cheilitis` | Cheilitis (Lip inflammation) |
| `Oral_Thrush` | Oral Candidiasis (Thrush) |
| `Nail_Psoriasis` | Nail Psoriasis |
| `Onychomycosis` | Onychomycosis (Fungal Nail Infection) |
| `Healthy_Skin` | Healthy/Normal Skin |

**Zero-Shot Fallback:** If no supervised head is available, BiomedCLIP performs **zero-shot text-image similarity** using clinical text prompts like *"a clinical image showing melanoma"*.

**Training details:**
- Optimizer: Adam (lr=0.001), Epochs: 25, Batch: 16
- Loss: Cross-Entropy
- Features are extracted from frozen BiomedCLIP backbone, only the head is trained

---

### 4.3 — `RadDINOService` (Radiology Scan Classifier)
**File:** `backend/app/services/models.py` → `RadDINOService` class  
**Foundation model:** `facebookresearch/dinov2_vits14` (Facebook AI / Meta)  
**Trained head weights:** `backend/models/dinov2_head.pth` (197 KB)  
**Training script:** `backend/app/scripts/train_classifiers.py`

**What it does:**  
Classifies radiology scans (X-ray, MRI, CT) into **6 pathology categories** using **DINOv2 ViT-S/14** — a self-supervised Vision Transformer pre-trained using DINO v2 distillation on 142 million images without labels. DINOv2 is known to produce highly discriminative visual features suited for medical imaging tasks.

> **Note:** The naming "RadDINO" is inspired by Microsoft's Rad-DINO model (a DINOv2 variant fine-tuned on radiology), but this project uses DINOv2 ViT-S/14 as the backbone with a custom supervised linear probe head.

**Foundation Model Details:**
- Architecture: ViT-S/14 (Vision Transformer Small, 14×14 patch, 224×224 input)
- Pre-training: DINO v2 self-supervised distillation on 142M images (LVD-142M dataset)
- Output embedding: 384-dimensional CLS token feature vector
- Input normalization: ImageNet mean/std [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]

**Supervised Head (LinearProbeClassifier):**
```
Input: 384-dim DINOv2 CLS embedding
  Linear(384→128) → ReLU → Dropout(0.2) → Linear(128→6)
Output: Softmax → class probabilities over 6 scan pathology classes
```

**Scan Disease Classes (6 classes):**
| Class Folder | Scan Type | Pathology |
|---|---|---|
| `Pneumonia` | Chest X-ray | Pneumonia |
| `Atelectasis` | Chest X-ray | Atelectasis (lung collapse) |
| `Brain_Tumor` | Brain MRI | Glioma / Brain Tumor |
| `Brain_Hemorrhage` | Brain MRI | Brain Hemorrhage |
| `Spine_Herniated_Disc` | MRI Spine | Herniated Disc |
| `Normal_Scan` | Any | Normal / Healthy scan |

**Training details:**
- Optimizer: Adam (lr=0.001), Epochs: 25, Batch: 16
- Loss: Cross-Entropy
- DINOv2 backbone is frozen; only linear probe head is fine-tuned

---

### 4.4 — `YOLOv11SegService` (Lesion Detection + Segmentation)
**File:** `backend/app/services/models.py` → `YOLOv11SegService` class  
**Model weights:** `backend/models/yolo11n.pt` (5.4 MB) + `backend/models/sam2_t.pt` (74.4 MB)  
**Training script:** `backend/app/scripts/train_yolo.py`

**What it does:**  
A **hybrid two-pass pipeline** combining YOLOv11 object detection with SAM 2 (Segment Anything Model 2) to localize the disease region and generate a pixel-level segmentation mask overlay on the uploaded image.

**Pass 1 — YOLOv11 (Bounding Box Detection):**
- Architecture: YOLOv11 Nano (`yolo11n.pt`) — Ultralytics' latest YOLO generation
- Purpose: Detect the most prominent object region in the image and return [xmin, ymin, xmax, ymax] bounding box coordinates
- If YOLO detects no confident box, a clinical center-region bounding box is computed using a filename hash

**Pass 2 — SAM 2 (Segment Anything Model 2):**
- Architecture: SAM 2 Tiny (`sam2_t.pt`) — Meta AI's SAM 2 model
- Purpose: Given the bounding box from Pass 1 as a prompt, SAM 2 generates a precise pixel-level binary mask of the region of interest
- The mask is rendered as a **semi-transparent red overlay (RGBA: 239, 68, 68, alpha=120)** saved as a PNG
- Output: `mask_{filename}.png` served alongside the original image

**Output:**
```json
{
  "bbox": [xmin, ymin, xmax, ymax],
  "mask_filename": "mask_<uuid>.png",
  "affected_pixel_count": 12345,
  "total_pixel_count": 262144
}
```

---

### 4.5 — Severity Estimator
**File:** `backend/app/services/severity.py`

**What it does:**  
A rule-based, disease-aware severity classifier that combines bounding box area ratio and segmented pixel mask ratio to compute an **involvement ratio**, then maps it to **Mild / Moderate / Severe** using clinical disease priors.

**Logic:**
```
involvement_ratio = max(bbox_area/total_pixels, mask_pixels/total_pixels)
```

| Disease Category | Mild | Moderate | Severe |
|---|---|---|---|
| Melanoma | — | Default | ratio > 35% |
| Basal Cell Carcinoma | — | Default | ratio > 40% |
| Eczema / Psoriasis | < 25% | 25–50% | > 50% |
| Burns | 1st degree | 2nd degree / > 20% | 3rd degree / > 45% |
| Wounds / Ulcers | < 20% | 20–40% | > 40% |
| Eye / Ophthalmology | Default Mild | Cataracts / > 30% | — |
| Chest / Pneumonia | < 25% | 25–45% | > 45% |
| Brain Tumor / Hemorrhage | — | Default | > 30% |
| Bone Fracture | — | Default | > 35% |

---

### 4.6 — Specialist Recommender
**File:** `backend/app/services/specialist.py`

Maps detected disease/body part combinations to the relevant specialist:

| Condition | Specialist |
|---|---|
| Skin cancer, Eczema, Psoriasis | Dermatologist |
| Third-degree burns | Burn Surgeon / Plastic Surgeon |
| Wounds, Ulcers | Wound Care Specialist |
| Eye disease | Ophthalmologist |
| Ear (Otitis) | Otolaryngologist (ENT) |
| Oral / Lip disease | Dentist / Oral Medicine Specialist |
| Pneumonia / Lung | Pulmonologist |
| Brain Tumor / Hemorrhage | Neurosurgeon / Oncologist |
| Herniated Disc | Orthopedic Spine Specialist |
| Fractures | Orthopedic Surgeon |
| Default Radiology | Radiologist / General Physician |

---

### 4.7 — RAG Chat Assistant
**File:** `backend/app/services/rag.py`

**What it does:**  
A Retrieval-Augmented Generation (RAG) system that allows the clinician/patient to ask follow-up questions about their case report. It retrieves the most relevant clinical guideline from a curated local vector database, combines it with patient context, and queries an LLM for a structured advisory response.

**Components:**

1. **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (local)  
   - Encodes clinical guideline texts and user queries into dense 384-dim vectors
   - Cosine similarity retrieval to find the most relevant guideline

2. **Knowledge Base:** `backend/app/data/medical_guidelines.json`  
   Pre-seeded with 8 clinical conditions: Melanoma, Eczema, Pneumonia, Burns, Fractures, Conjunctivitis, Paronychia, Pressure Ulcers

3. **LLM Provider (Configurable):**
   - **Gemini Pro** (Google) — via REST API
   - **GPT-4 Turbo** (OpenAI) — via REST API
   - **Local Ollama** — `qwen2.5:1.5b` or `llama3.1` running locally
   - **Heuristic Fallback** — rule-based template if all LLMs are offline

4. **Output:** Natural language advice + structured checklist `[ ] action item`

---

### 4.8 — PDF Report Generator
**File:** `backend/app/services/report.py`

Generates a premium-styled clinical PDF using **ReportLab** with:
- Patient demographics table
- AI pipeline findings table (modality, disease, confidence, severity, specialist)
- Clinical summary narrative
- RAG-generated suggestions as bullet points
- Medical disclaimer footer
- Color-coded severity (Teal=Normal, Amber=Moderate, Red=Severe)

---

## 5. Training Data

### Total Training Images: **2,590**

#### RGB Branch (BiomedCLIP Supervised Head) — 1,284 images across 9 classes:

| Class | Images |
|---|---|
| Eczema | 204 |
| Onychomycosis (Fungal Nail) | 193 |
| Oral Thrush | 177 |
| Psoriasis | 152 |
| Nail Psoriasis | 123 |
| Healthy Skin | 114 |
| Melanoma | 116 |
| Basal Cell Carcinoma | 113 |
| Cheilitis | 92 |
| **Total** | **1,284** |

#### Scan Branch (DINOv2 Supervised Head) — 1,306 images across 6 classes:

| Class | Scan Type | Images |
|---|---|---|
| Normal Scan | Any | 302 |
| Spine Herniated Disc | MRI Spine | 240 |
| Atelectasis | Chest X-ray | 204 |
| Pneumonia | Chest X-ray | 200 |
| Brain Tumor | Brain MRI | 192 |
| Brain Hemorrhage | Brain MRI | 168 |
| **Total** | | **1,306** |

**Data format:** Images stored in class-named subfolders under `backend/app/data/supervised/RGB/` and `backend/app/data/supervised/Scan/`  
**Image types:** `.png`, `.jpg`, `.jpeg`, `.webp`

> **Note:** The Modality Router CNN also trains on ALL 2,590 images (labeled as RGB=1 or Scan=0) for the binary classification task.

---

## 6. Technology Stack

### Backend
| Component | Technology | Version |
|---|---|---|
| API Framework | FastAPI | 0.111.0 |
| ASGI Server | Uvicorn | 0.30.1 |
| Data Validation | Pydantic v2 | 2.7.4 |
| Primary Database | MongoDB (Motor async driver) | pymongo 4.7.3 / motor 3.4.0 |
| Fallback Database | SQLite (built-in) | — |
| Deep Learning | PyTorch | 2.3.1 |
| Computer Vision | TorchVision | 0.18.1 |
| YOLO + SAM 2 | Ultralytics | 8.2.32 |
| CLIP / BiomedCLIP | open-clip-torch | 2.24.0 |
| RAG Embeddings | sentence-transformers | 3.0.1 |
| PDF Generation | ReportLab | 4.2.0 |
| Image Processing | Pillow | 10.3.0 |
| Numerical Computing | NumPy | 1.26.4 |
| HTTP Client (LLM) | Requests | 2.32.3 |
| Config / Env | python-dotenv | 1.0.1 |
| Templating | Jinja2 | 3.1.4 |

### Frontend
| Component | Technology | Version |
|---|---|---|
| UI Framework | React 19 | ^19.2.7 |
| Build Tool | Vite | ^8.1.1 |
| Styling | Vanilla CSS (custom design system) | — |
| Linting | OxLint | ^1.71.0 |

### AI Models Used
| Model | Source | Purpose |
|---|---|---|
| BiomedCLIP ViT-B/16 | Microsoft / HuggingFace | RGB disease classification |
| DINOv2 ViT-S/14 | Facebook AI (PyTorch Hub) | Radiology scan classification |
| YOLOv11 Nano | Ultralytics | Bounding box detection |
| SAM 2 Tiny | Meta AI / Ultralytics | Pixel-level segmentation mask |
| all-MiniLM-L6-v2 | sentence-transformers | RAG guideline embedding |
| Gemini Pro / GPT-4 / Ollama | Google / OpenAI / Local | LLM advisory generation |

---

## 7. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/analysis/upload` | Upload image + patient info → full AI pipeline analysis |
| GET | `/api/reports/` | Fetch all previous patient reports |
| GET | `/api/reports/{id}` | Fetch single report by ID |
| GET | `/api/reports/{id}/pdf` | Download PDF report |
| POST | `/api/chat/query` | RAG chat query against a specific report |
| GET | `/uploads/{filename}` | Serve uploaded images and mask overlays (static) |

---

## 8. Frontend Features

The frontend is a single-page React application with the following views/tabs:

1. **Dashboard Tab** — Statistics cards (total cases, active cases, critical cases) + recent reports table
2. **New Analysis Tab** — Patient intake form (name, age, sex, symptoms, history) + drag-and-drop image upload + real-time AI processing pipeline with animated status steps
3. **Results View** — Side-by-side original image + segmentation mask overlay toggle, bounding box coordinates, disease classification badge, severity indicator, specialist recommendation, full report text, RAG suggestions checklist, PDF download button
4. **AI Advisor Chat** — Real-time RAG chat interface tied to the active case report; conversation history maintained in-session

**UI Design System:**
- Dark theme glassmorphism design
- Mouse-following spotlight glow effect (lerp-smoothed)
- Scroll-triggered reveal animations (IntersectionObserver)
- Scroll progress indicator
- Animated gradient background

---

## 9. What We Are Further Including / Future Scope

Based on the project structure and training scripts, the following are planned or infrastructure is already in place for:

1. **Custom YOLOv11-Seg Training** — `train_yolo.py` sets up a complete YOLO segmentation dataset structure with 21 disease classes for training a domain-specific YOLO-Seg model (currently using base weights). Infrastructure: `backend/app/data/yolo_dataset/`

2. **Expanded Disease Classes** — YOLO config includes 21 target classes:
   - Melanoma, Basal_Cell_Carcinoma, Eczema, Psoriasis, Conjunctivitis, Stye, Blepharitis, Paronychia, Finger_Fracture, Cut_Laceration, Onychomycosis, Diabetic_Ulcer, Pressure_Ulcer, Surgical_Incision, Burn_Wound, Pneumonia, Atelectasis, Pleural_Effusion, Brain_Tumor, Brain_Hemorrhage, Spine_Herniated_Disc

3. **Expanded Clinical Guidelines Database** — The JSON knowledge base (`medical_guidelines.json`) can be extended with more conditions beyond the current 8 entries

4. **LLM-Enhanced Report Generation** — The `report_text` field can be replaced with full LLM-generated narrative using the same Gemini/OpenAI/Ollama stack

5. **Zero-Shot Localization** — `zero_shot_localize.py` is scaffolded for gradient-based localization (GradCAM / attention map) to highlight regions without YOLO boxes

6. **More Specialist Branches** — Dermatoscopy, fundus photography, oral pathology, bone densitometry

7. **User Authentication / Multi-user** — Patient case management with clinic-level authentication

---

## 10. Project Folder Structure

```
IV-I/
├── backend/
│   ├── app/
│   │   ├── main.py              ← FastAPI app entry point
│   │   ├── config.py            ← Env-based configuration
│   │   ├── database.py          ← MongoDB + SQLite fallback DB layer
│   │   ├── routers/
│   │   │   ├── analysis.py      ← /api/analysis/upload endpoint
│   │   │   ├── reports.py       ← /api/reports/* endpoints
│   │   │   └── chat.py          ← /api/chat/query RAG endpoint
│   │   ├── services/
│   │   │   ├── models.py        ← All 4 AI model classes (CNN, CLIP, DINOv2, YOLO+SAM)
│   │   │   ├── pipeline.py      ← Orchestrates the full 5-step AI pipeline
│   │   │   ├── severity.py      ← Disease-aware severity classification
│   │   │   ├── specialist.py    ← Specialist recommendation lookup
│   │   │   ├── rag.py           ← RAG retrieval + LLM advisory system
│   │   │   └── report.py        ← PDF report generation (ReportLab)
│   │   ├── scripts/
│   │   │   ├── train_modality.py     ← Train SimpleModalityCNN
│   │   │   ├── train_classifiers.py  ← Train BiomedCLIP + DINOv2 linear heads
│   │   │   ├── train_yolo.py         ← YOLO-Seg fine-tuning scaffold
│   │   │   ├── zero_shot_localize.py ← GradCAM localization
│   │   │   └── clear_db.py           ← DB management utility
│   │   └── data/
│   │       ├── supervised/
│   │       │   ├── RGB/          ← 1,284 visible-light disease images (9 classes)
│   │       │   └── Scan/         ← 1,306 radiology scan images (6 classes)
│   │       └── medical_guidelines.json  ← RAG knowledge base
│   ├── models/
│   │   ├── modality_model.pth         ← SimpleModalityCNN weights (8.1 MB)
│   │   ├── biomed_clip_head.pth       ← BiomedCLIP linear probe head (263 KB)
│   │   ├── biomed_clip_classes.json   ← 9 RGB class labels (serialized)
│   │   ├── dinov2_head.pth            ← DINOv2 linear probe head (197 KB)
│   │   ├── dinov2_classes.json        ← 6 Scan class labels (serialized)
│   │   ├── yolo11n.pt                 ← YOLOv11 Nano detector weights (5.4 MB)
│   │   ├── yolo11n-seg.pt             ← YOLOv11 Nano-Seg weights (5.9 MB)
│   │   └── sam2_t.pt                  ← SAM 2 Tiny weights (74.4 MB)
│   ├── uploads/                       ← Runtime uploaded images + mask overlays
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx              ← Full React SPA (693 lines)
│   │   ├── index.css            ← Full custom CSS design system
│   │   └── main.jsx
│   ├── index.html
│   └── package.json
└── Vision-Language_Models_in_Medical_Applications_A_Review.pdf  ← Base paper
```

---

## 11. Key Design Decisions

| Decision | Rationale |
|---|---|
| **Dual-branch architecture** (BiomedCLIP for RGB, DINOv2 for Scans) | Each foundation model is optimized for its domain: BiomedCLIP is pre-trained on biomedical text-image pairs; DINOv2 produces excellent spatial features for medical scans |
| **Linear Probe (not full fine-tuning)** | Prevents overfitting on a limited dataset; leverages rich pre-trained representations; computationally efficient |
| **Hybrid YOLO + SAM 2** | YOLO provides fast object-level detection; SAM 2 provides precise pixel-level segmentation; combining them gives the best of both worlds |
| **MongoDB + SQLite fallback** | Allows deployment without a cloud database; SQLite auto-activates if MongoDB is unavailable |
| **Multi-provider LLM (Gemini / OpenAI / Ollama)** | Flexibility for offline (local Ollama), cloud (Gemini/OpenAI), and free-tier usage |
| **Severity from pixel ratios** | Disease involvement area (mask ratio) is a clinically interpretable proxy for severity used in dermatology and radiology literature |

---

*Document auto-generated from full source code analysis — July 2026*
