# Medical AI Pipeline: Machine Learning Models & Training Documentation

This document compiles the architectural specifications, training workflows, and deployment procedures for the machine learning models integrated into the **Intelligent Medical Image Analysis with Automated Clinical Report Generation** system.

---

## 1. Modality Router CNN (Grayscale Scans vs. Color Photos)

The Modality Router is the entry gate of the pipeline. It classifies incoming images as either **RGB (visible light body parts)** or **Scan (grayscale radiology imagery like CT, MRI, X-rays)**.

### Architecture Specifications
* **Network Type:** Custom 3-Layer Convolutional Neural Network (ConvNet) built in PyTorch.
* **Input Resolution:** $128 \times 128 \times 3$
* **Output Classes:** `[Scan, RGB]` (Softmax Logits)
* **Convolutional Layers:**
  1. `Conv2d(3, 16, kernel_size=3, padding=1)` + `ReLU` + `MaxPool2d(2, 2)` (Reduces spatial dimension to $64 \times 64$)
  2. `Conv2d(16, 32, kernel_size=3, padding=1)` + `ReLU` + `MaxPool2d(2, 2)` (Reduces spatial dimension to $32 \times 32$)
  3. `Conv2d(32, 64, kernel_size=3, padding=1)` + `ReLU` + `MaxPool2d(2, 2)` (Reduces spatial dimension to $16 \times 16$)
* **Classifier Head:**
  - `Flatten()` $\rightarrow$ `Linear(64 * 16 * 16, 128)` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Linear(128, 2)`

### Training Procedure
A training utility script is available at `backend/app/scripts/train_modality.py`.
1. Run dataset collection. Organize sample images under:
   - `backend/app/data/modality/train/rgb/` and `scan/`
   - `backend/app/data/modality/val/rgb/` and `scan/`
2. Run training:
   ```bash
   python backend/app/scripts/train_modality.py
   ```
3. The script executes training using Adam optimizer, computes Cross-Entropy loss, and saves weights to `backend/models/modality_model.pth`.
4. The API server dynamically loads this model on startup and executes PyTorch predictions on CUDA/CPU.

---

## 2. Zero-Shot BiomedCLIP (RGB Body Parts & Disease Classifier)

BiomedCLIP is a medical vision-language foundation model trained by Microsoft on 15 million biomedical image-text pairs (PMC-15M).

### Encoders Structure
* **Vision Encoder:** Vision Transformer (ViT-B/16) pre-trained on medical imagery. Projecting global `[CLS]` tokens into a shared $512$-dimensional space.
* **Text Encoder:** PubMedBERT pre-trained on medical abstracts, matching tokens into the same $512$-dimensional space.

### Zero-Shot Pipeline Setup
To classify without training dataset heads:
1. Extract $512$-dimensional image features:
   $$E_{\text{img}} = \text{ViT-Encoder}(\text{Image})$$
2. Tokenize and extract text prompts:
   $$E_{\text{txt}} = \text{PubMedBERT-Encoder}(\text{"a clinical photo of a skin lesion"})$$
3. Compute Cosine Similarity between normalized embeddings:
   $$\text{Similarity}(I, T_i) = \frac{E_{\text{img}}(I) \cdot E_{\text{txt}}(T_i)}{\|E_{\text{img}}(I)\| \|E_{\text{txt}}(T_i)\|} \times e^{\tau}$$
4. Identify the highest similarity prompt. This gives zero-shot body part detection. A second pass matches the corresponding zero-shot diseases.

---

## 3. Radiology Scan Classifier (DINOv2 / Rad-DINO)

For grayscale scans (X-ray, CT, MRI), we load the self-supervised **DINOv2 (ViT-Small/14)** model via PyTorch Hub.

### Features Extraction
* **Global Embeddings:** The global `[CLS]` token returns a $384$-dimensional feature vector representing global scan statistics and organs density layout.
* **Local Patch Tokens:** Returns $256 \times 384$ local tokens (corresponding to $14 \times 14$ grid patches).
* **classification Head:** If pre-computed radiology vectors are loaded, similarity matching classifies the scan type (Chest X-ray vs Brain MRI) and detects diseases.

---

## 4. Bounding Box & Segmentation: YOLOv11-Seg

YOLOv11-Seg is the real-time localization engine. It runs locally using the `ultralytics` package.

### Custom Model Training Workflow
To train YOLOv11-Seg to detect and segment custom lesions/abnormalities:
1. **Prepare Dataset Directory:**
   ```text
   data/yolo_dataset/
     images/
       train/ (img1.jpg, img2.jpg...)
       val/
     labels/
       train/ (img1.txt, img2.txt...) - matching image filenames
       val/
   ```
2. **Annotation Format:**
   Each label `.txt` file must contain normalized polygon boundaries relative to image width/height:
   `class_id x1 y1 x2 y2 x3 y3 ... xN yN`
3. **Execute YOLOv11 Training:**
   Execute the template at `backend/app/scripts/train_yolo.py`:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov11n-seg.pt")
   model.train(data="data/yolo_dataset/dataset.yaml", epochs=50, imgsz=640, device="cuda")
   ```

### Pre-trained Zero-Shot Alternative (No Training Data)
For cases where annotated segment masks are missing, we recommend integrating **Grounding DINO + SAM 2**:
* **Grounding DINO:** Detects bounding boxes zero-shot based on text tags (e.g. *"dark lesion on skin"*, *"fracture in finger bone"*).
* **Segment Anything 2 (SAM 2):** Takes bounding boxes from Grounding DINO and segments boundaries of the pathology in zero-shot mode.
* Execution code and package dependencies are documented in `backend/app/scripts/zero_shot_localize.py`.

---

## 5. Live RAG Guidance & LLM Connections

The RAG Advisor leverages vector guideline similarity queries and a clinical LLM generator:

1. **Local Vector Database:**
   - Seeded in `backend/app/data/medical_guidelines.json`.
   - Embeds query and database texts using `sentence-transformers/all-MiniLM-L6-v2`.
   - Computes cosine similarity to retrieve matching clinical evidence.
2. **LLM Connection Setup:**
   Create a `.env` file in `backend/app/` using the `.env.example` template:
   - Add your `GEMINI_API_KEY` to connect directly to Gemini Pro.
   - Or start a local **Ollama** server running `ollama run qwen2.5:1.5b`.
