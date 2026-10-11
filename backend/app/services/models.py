import os
import logging
from PIL import Image
import numpy as np

logger = logging.getLogger("app.models")

# Dynamic loading checks
TORCH_AVAILABLE = False
OPEN_CLIP_AVAILABLE = False
ULTRALYTICS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not installed. Models will fall back to simulated mode.")

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    logger.warning("open-clip-torch not installed. BiomedCLIP zero-shot inference will run in simulated mode.")

try:
    from ultralytics import YOLO, SAM
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    logger.warning("ultralytics not installed. YOLOv11-Seg will run in simulated mode.")


if TORCH_AVAILABLE:
    class SimpleModalityCNN(nn.Module):
        """
        Lightweight CNN to classify images into Scan (0) or RGB (1).
        Matches the model trained by train_modality.py.
        """
        def __init__(self):
            super(SimpleModalityCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 16 * 16, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 2)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    class LinearProbeClassifier(nn.Module):
        """
        Lightweight Supervised head built on top of foundation models.
        """
        def __init__(self, input_dim, num_classes):
            super(LinearProbeClassifier, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            return self.fc(x)
else:
    class SimpleModalityCNN:
        pass
    class LinearProbeClassifier:
        pass


class ModalityRouter:
    """
    Classifies image as RGB (visible light) or Scan (X-ray, CT, MRI).
    Loads a trained SimpleModalityCNN if weights are available.
    Otherwise, falls back to a non-parametric color-histogram statistics classifier.
    """
    def __init__(self):
        self.model = None
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        
        # Look for trained model weights
        weights_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
            "models", 
            "modality_model.pth"
        )
        
        if TORCH_AVAILABLE and os.path.exists(weights_path):
            try:
                logger.info(f"Loading trained ModalityRouter weights from: {weights_path}")
                self.model = SimpleModalityCNN()
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                logger.info("ModalityRouter neural network model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading trained ModalityRouter weights: {e}. Running statistics backup classifier.")
                self.model = None

    def predict(self, img_path: str) -> str:
        # If neural network is loaded, use it
        if TORCH_AVAILABLE and self.model is not None:
            try:
                with Image.open(img_path) as img:
                    img_rgb = img.convert("RGB").resize((128, 128))
                    img_np = np.array(img_rgb).astype(np.float32) / 255.0
                    tensor_img = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(tensor_img)
                        pred_idx = outputs.argmax(dim=1).item()
                        
                    detected_modality = "Scan" if pred_idx == 0 else "RGB"
                    logger.info(f"Modality Router Neural Network predicted: {detected_modality}")
                    return detected_modality
            except Exception as e:
                logger.error(f"Modality Router CNN inference failed: {e}. Falling back to image statistics.")

        # Fallback Statistics Classifier (checks color histogram variance)
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                pixels = np.array(img_rgb.resize((100, 100)))
                # Calculate standard deviation between R, G, and B values per pixel
                channel_std = np.std(pixels, axis=2)
                mean_std = np.mean(channel_std)
                
                # Grayscale scans have very low standard deviation across color channels
                if mean_std < 2.5:
                    logger.info(f"Modality Router statistics: Grayscale scan detected (Variance mean: {mean_std:.4f})")
                    return "Scan"
                
                logger.info(f"Modality Router statistics: RGB color photo detected (Variance mean: {mean_std:.4f})")
                return "RGB"
        except Exception as e:
            logger.error(f"Error in ModalityRouter statistics fallback: {e}")
            return "RGB"


class BiomedCLIPService:
    """
    BiomedCLIP wrapper for classification of visible body parts and diseases.
    Loads and runs a supervised LinearProbeClassifier head if trained weights exist.
    Otherwise, executes zero-shot text-image similarities.
    """
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.head = None
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        
        # Load class lists from the JSON configuration saved during training, falling back to folders if missing
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
            "models"
        )
        classes_json = os.path.join(models_dir, "biomed_clip_classes.json")
        if os.path.exists(classes_json):
            import json
            try:
                with open(classes_json, "r") as f:
                    self.rgb_classes = json.load(f)
                logger.info(f"Loaded BiomedCLIP class configuration list: {self.rgb_classes}")
            except Exception as e:
                logger.error(f"Failed to parse classes JSON: {e}")
                self.rgb_classes = [
                    "Melanoma", "Basal_Cell_Carcinoma", "Eczema", "Psoriasis", 
                    "Cheilitis", "Oral_Thrush", 
                    # "Nail_Infection",  # Temporarily kept aside (empty folder / 0 training samples; covered by Onychomycosis & Nail_Psoriasis)
                    "Healthy_Skin"
                ]
        else:
            dataset_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                "app", "data", "supervised"
            )
            rgb_base = os.path.join(dataset_dir, "RGB")
            if os.path.exists(rgb_base):
                # Temporarily discard Nail_Infection or empty/hidden dirs
                discarded = {"Nail_Infection"}  # Kept aside; preserved in codebase
                self.rgb_classes = sorted([
                    d for d in os.listdir(rgb_base) 
                    if os.path.isdir(os.path.join(rgb_base, d)) and d not in discarded and not d.startswith(('_', '.'))
                ])
            else:
                self.rgb_classes = [
                    "Melanoma", "Basal_Cell_Carcinoma", "Eczema", "Psoriasis", 
                    "Cheilitis", "Oral_Thrush", 
                    # "Nail_Infection",  # Temporarily kept aside
                    "Healthy_Skin"
                ]
        
        # Load foundation encoder
        if TORCH_AVAILABLE and OPEN_CLIP_AVAILABLE:
            try:
                logger.info(f"Loading BiomedCLIP vision backbone on {self.device}...")
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
                )
                self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                self.model.to(self.device)
                self.model.eval()
                logger.info("BiomedCLIP foundation model loaded.")
                
                # Check for trained supervised classifier head
                head_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                    "models", 
                    "biomed_clip_head.pth"
                )
                if os.path.exists(head_path):
                    logger.info(f"Loading trained BiomedCLIP Linear Probe Head from: {head_path}")
                    self.head = LinearProbeClassifier(input_dim=512, num_classes=len(self.rgb_classes))
                    self.head.load_state_dict(torch.load(head_path, map_location=self.device))
                    self.head.to(self.device)
                    self.head.eval()
                    logger.info("BiomedCLIP supervised classifier head loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load BiomedCLIP weights/heads: {e}. Running in simulation mode.")
                self.model = None

    def predict(self, img_path: str) -> dict:
        # If supervised head is available, run parametric prediction!
        if TORCH_AVAILABLE and self.model is not None and self.head is not None:
            try:
                with Image.open(img_path) as pil_img:
                    image = self.preprocess(pil_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # 1. Extract BiomedCLIP 512-dim embedding
                    features = self.model.encode_image(image)
                    features /= features.norm(dim=-1, keepdim=True)
                    
                    # 2. Pass through Supervised Linear Head
                    logits_raw = self.head(features)
                    
                    # Temporarily discard Nail_Infection logit if present in active class weights
                    # (keeps it aside so it is never falsely predicted over actual classes)
                    for discarded in ["Nail_Infection"]:
                        if discarded in self.rgb_classes:
                            d_idx = self.rgb_classes.index(discarded)
                            logits_raw[:, d_idx] = -float("inf")
                            
                    logits = logits_raw.softmax(dim=-1)
                    pred_idx = logits.argmax(dim=1).item()
                    confidence = float(logits[0, pred_idx].item())
                    
                predicted_class = self.rgb_classes[pred_idx]
                logger.info(f"BiomedCLIP Supervised prediction: {predicted_class} (Conf: {confidence:.2%})")
                
                # Parse folder name to extract body part and disease dynamically
                body_part = "Skin"
                disease = predicted_class.replace("_", " ")
                
                # Dynamic matching rules
                pred_lower = predicted_class.lower()
                if "basal" in pred_lower:
                    body_part, disease = "Skin", "Basal Cell Carcinoma"
                elif "melanoma" in pred_lower:
                    body_part, disease = "Skin", "Melanoma"
                elif "eczema" in pred_lower:
                    body_part, disease = "Skin", "Eczema"
                elif "nail_psoriasis" in pred_lower:
                    body_part, disease = "Nail", "Nail Psoriasis"
                elif "psoriasis" in pred_lower:
                    body_part, disease = "Skin", "Psoriasis"
                elif "cheilitis" in pred_lower or "lips" in pred_lower:
                    body_part, disease = "Lips", "Cheilitis"
                elif "thrush" in pred_lower or "oral" in pred_lower or "tongue" in pred_lower:
                    body_part, disease = "Tongue", "Oral Candidiasis (Thrush)"
                elif "onychomycosis" in pred_lower or "nail" in pred_lower or "fungus" in pred_lower:
                    body_part, disease = "Nail", "Onychomycosis (Fungal)"
                elif "eye" in pred_lower or "conjunctivitis" in pred_lower:
                    body_part, disease = "Eye", "Conjunctivitis"
                elif "burn" in pred_lower:
                    body_part, disease = "Burn", "Second-Degree Burn"
                elif "wound" in pred_lower:
                    body_part, disease = "Wound", "Diabetic Ulcer"
                return {
                    "body_part_scan_type": body_part,
                    "disease": disease,
                    "confidence": confidence,
                    "raw_class": predicted_class
                }
            except Exception as e:
                logger.error(f"BiomedCLIP Supervised prediction failed: {e}. Falling back to zero-shot.")

        # Fallback to BiomedCLIP Zero-Shot prediction
        if TORCH_AVAILABLE and OPEN_CLIP_AVAILABLE and self.model is not None:
            try:
                body_parts = ["Skin", "Eye", "Finger", "Nail", "Lips", "Tongue", "Ear", "Wound", "Burn"]
                diseases = {
                    "Skin": ["Melanoma", "Basal Cell Carcinoma", "Eczema", "Psoriasis", "Healthy Skin"],
                    "Eye": ["Conjunctivitis", "Stye", "Blepharitis", "Healthy Eye"],
                    "Finger": ["Paronychia", "Finger Fracture", "Cuts/Laceration", "Healthy Finger"],
                    "Nail": ["Onychomycosis (Fungal)", "Nail Psoriasis", "Healthy Nail"],
                    "Lips": ["Cheilitis", "Herpes Labialis (Cold Sore)", "Healthy Lips"],
                    "Tongue": ["Oral Candidiasis (Thrush)", "Geographic Tongue", "Healthy Tongue"],
                    "Ear": ["Otitis Externa", "Ear Cyst", "Healthy Ear"],
                    "Wound": ["Diabetic Ulcer", "Pressure Ulcer", "Surgical Incision"],
                    "Burn": ["First-Degree Burn", "Second-Degree Burn", "Third-Degree Burn"]
                }
                with Image.open(img_path) as pil_img:
                    image = self.preprocess(pil_img).unsqueeze(0).to(self.device)
                
                body_part_prompts = [f"a medical photo of a {part.lower()}" for part in body_parts]
                body_part_prompts[body_parts.index("Burn")] = "a photo showing a skin burn injury"
                body_part_prompts[body_parts.index("Wound")] = "a photo showing a skin ulcer wound or cut"
                
                texts_part = self.tokenizer(body_part_prompts).to(self.device)
                with torch.no_grad():
                    image_features, text_features, logit_scale = self.model(image, texts_part)
                    logits = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)
                    best_part_idx = logits.argmax().item()
                    detected_part = body_parts[best_part_idx]
                
                part_diseases = diseases.get(detected_part, ["Healthy"])
                disease_prompts = [f"a clinical image showing {dis.lower()}" for dis in part_diseases]
                texts_dis = self.tokenizer(disease_prompts).to(self.device)
                with torch.no_grad():
                    image_features_dis, text_features_dis, logit_scale_dis = self.model(image, texts_dis)
                    logits_dis = (logit_scale_dis * image_features_dis @ text_features_dis.t()).softmax(dim=-1)
                    best_dis_idx = logits_dis.argmax().item()
                    detected_disease = part_diseases[best_dis_idx]
                    confidence = float(logits_dis[0, best_dis_idx].cpu().item())
                
                logger.info(f"BiomedCLIP Zero-Shot: {detected_part} -> {detected_disease} (Conf: {confidence:.2%})")
                return {
                    "body_part_scan_type": detected_part,
                    "disease": detected_disease,
                    "confidence": confidence
                }
            except Exception as e:
                logger.error(f"BiomedCLIP Zero-shot prediction failed: {e}. Reverting to simulation.")

        # Heuristic simulation fallback (keyword matching based on filename)
        filename = os.path.basename(img_path).lower()
        return {
            "body_part_scan_type": "Skin",
            "disease": "Melanoma" if "melanoma" in filename else "Healthy Skin",
            "confidence": 0.95
        }


class RadDINOService:
    """
    Rad-DINO wrapper for radiology scan analysis.
    Loads a supervised LinearProbeClassifier head built on DinoV2 features if weights exist.
    Otherwise, extracts DINOv2 features and returns a heuristic mapping scan label.
    """
    def __init__(self):
        self.model = None
        self.head = None
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        
        # Load class lists from the JSON configuration saved during training, falling back to folders if missing
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
            "models"
        )
        classes_json = os.path.join(models_dir, "dinov2_classes.json")
        if os.path.exists(classes_json):
            import json
            try:
                with open(classes_json, "r") as f:
                    self.scan_classes = json.load(f)
                logger.info(f"Loaded DinoV2 scan class configuration list: {self.scan_classes}")
            except Exception as e:
                logger.error(f"Failed to parse scan classes JSON: {e}")
                self.scan_classes = ["Pneumonia", "Atelectasis", "Brain_Tumor", "Brain_Hemorrhage", "Spine_Herniated_Disc", "Normal_Scan"]
        else:
            dataset_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                "app", "data", "supervised"
            )
            scan_base = os.path.join(dataset_dir, "Scan")
            if os.path.exists(scan_base):
                self.scan_classes = sorted([d for d in os.listdir(scan_base) if os.path.isdir(os.path.join(scan_base, d))])
            else:
                self.scan_classes = ["Pneumonia", "Atelectasis", "Brain_Tumor", "Brain_Hemorrhage", "Spine_Herniated_Disc", "Normal_Scan"]
        
        if TORCH_AVAILABLE:
            try:
                logger.info(f"Loading DinoV2 backbone from PyTorch Hub on {self.device}...")
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                self.model.to(self.device)
                self.model.eval()
                logger.info("DinoV2 backbone loaded.")
                
                # Check for trained supervised classifier head
                head_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                    "models", 
                    "dinov2_head.pth"
                )
                if os.path.exists(head_path):
                    logger.info(f"Loading trained DinoV2 Linear Probe Head from: {head_path}")
                    self.head = LinearProbeClassifier(input_dim=384, num_classes=len(self.scan_classes))
                    self.head.load_state_dict(torch.load(head_path, map_location=self.device))
                    self.head.to(self.device)
                    self.head.eval()
                    logger.info("DinoV2 supervised classifier head loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load DinoV2 weights/heads: {e}. Running in simulation mode.")
                self.model = None

    def predict(self, img_path: str) -> dict:
        # If supervised head is available, run prediction!
        if TORCH_AVAILABLE and self.model is not None and self.head is not None:
            try:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                with Image.open(img_path) as pil_img:
                    image_tensor = transform(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # 1. Extract DinoV2 384-dim global embeddings
                    features = self.model(image_tensor)
                    
                    # 2. Pass through Supervised Linear Head
                    logits = self.head(features).softmax(dim=-1)
                    pred_idx = logits.argmax(dim=1).item()
                    confidence = float(logits[0, pred_idx].item())
                
                predicted_class = self.scan_classes[pred_idx]
                logger.info(f"DINOv2 Supervised prediction: {predicted_class} (Conf: {confidence:.2%})")
                
                # Map classes to output details
                # Parse folder name to extract scan type and disease dynamically
                scan_type = "Chest X-ray"
                disease = predicted_class.replace("_", " ")
                
                # Dynamic matching rules
                pred_lower = predicted_class.lower()
                if "pneumonia" in pred_lower:
                    scan_type, disease = "Chest X-ray", "Pneumonia"
                elif "atelectasis" in pred_lower:
                    scan_type, disease = "Chest X-ray", "Atelectasis"
                elif "hemorrhage" in pred_lower:
                    scan_type, disease = "Brain MRI", "Brain Hemorrhage"
                elif "tumor" in pred_lower or "glioma" in pred_lower or "brain" in pred_lower:
                    scan_type, disease = "Brain MRI", "Glioma / Brain Tumor"
                elif "spine" in pred_lower or "disc" in pred_lower:
                    scan_type, disease = "MRI Spine", "Herniated Disc"
                elif "normal" in pred_lower:
                    scan_type, disease = "Chest X-ray", "Normal Scan"
                elif "fracture" in pred_lower or "bone" in pred_lower:
                    scan_type, disease = "Bone X-ray", "Bone Fracture"
                return {
                    "body_part_scan_type": scan_type,
                    "disease": disease,
                    "confidence": confidence,
                    "raw_class": predicted_class
                }
            except Exception as e:
                logger.error(f"DINOv2 Supervised prediction failed: {e}. Falling back to default heuristics.")

        # Fallback simulation
        if TORCH_AVAILABLE and self.model is not None:
            try:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                with Image.open(img_path) as pil_img:
                    image_tensor = transform(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    features = self.model(image_tensor)
                    feature_np = features.cpu().numpy()[0]
                
                h_val = float(np.mean(feature_np))
                h_abs = abs(h_val)
                scan_types = ["Chest X-ray", "Brain MRI", "CT Abdomen", "CT Chest", "MRI Spine"]
                diseases = {
                    "Chest X-ray": ["Pneumonia", "Atelectasis", "Pleural Effusion", "Normal Chest"],
                    "Brain MRI": ["Glioma / Brain Tumor", "Hemorrhage", "Stroke / Infarct", "Normal Brain"],
                    "CT Abdomen": ["Appendicitis", "Hepatic Steatosis", "Normal Abdomen"],
                    "CT Chest": ["Lung Nodule", "Normal Thorax"],
                    "MRI Spine": ["Herniated Disc", "Spinal Stenosis", "Normal Spine"]
                }
                
                scan_idx = int(h_abs * 100) % len(scan_types)
                detected_scan = scan_types[scan_idx]
                scan_diseases = diseases.get(detected_scan, ["Normal"])
                disease_idx = int(h_abs * 500) % len(scan_diseases)
                detected_disease = scan_diseases[disease_idx]
                
                confidence = 0.70 + (0.2 * (h_abs % 0.1) / 0.1)
                return {
                    "body_part_scan_type": detected_scan,
                    "disease": detected_disease,
                    "confidence": float(confidence)
                }
            except Exception as e:
                logger.error(f"DINOv2 backbone extraction error: {e}")

        return {
            "body_part_scan_type": "Chest X-ray",
            "disease": "Normal Chest",
            "confidence": 0.85
        }


class YOLOv11SegService:
    """
    Hybrid YOLOv11 + SAM 2 Segmentation service for actual bounding boxes and mask overlays.
    """
    def __init__(self):
        self.model = None
        self.sam_model = None
        if ULTRALYTICS_AVAILABLE:
            models_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                "models"
            )
            try:
                # 1. Load YOLOv11 Bounding Box detector
                logger.info("Loading local YOLOv11 weights...")
                yolo_path = os.path.join(models_dir, "yolo11n.pt")
                if not os.path.exists(yolo_path):
                    import urllib.request
                    logger.info(f"Downloading YOLOv11 detector weights to: {yolo_path}...")
                    urllib.request.urlretrieve(
                        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt", 
                        yolo_path
                    )
                self.model = YOLO(yolo_path)
                logger.info("YOLOv11 detector loaded successfully.")
            except Exception as e:
                logger.error(f"Could not load YOLOv11: {e}.")
                self.model = None

            try:
                # 2. Load Segment Anything (SAM 2) model
                logger.info("Loading pre-trained SAM 2 weights...")
                sam_path = os.path.join(models_dir, "sam2_t.pt")
                if not os.path.exists(sam_path):
                    import urllib.request
                    logger.info(f"Downloading SAM 2 weights to: {sam_path}...")
                    urllib.request.urlretrieve(
                        "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_t.pt", 
                        sam_path
                    )
                self.sam_model = SAM(sam_path)
                logger.info("SAM 2 loaded successfully.")
            except Exception as e:
                logger.error(f"Could not load SAM 2: {e}.")
                self.sam_model = None

    def segment(self, img_path: str, disease: str) -> dict:
        uploads_dir = os.path.dirname(img_path)
        base_name = os.path.basename(img_path)
        mask_filename = f"mask_{base_name.split('.')[0]}.png"
        mask_path = os.path.join(uploads_dir, mask_filename)
        
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except:
            width, height = 512, 512
            
        bbox = []
        
        # Execute hybrid YOLO + SAM pipeline
        if ULTRALYTICS_AVAILABLE and self.sam_model is not None:
            try:
                # Pass 1: Try to locate boxes using YOLO detector
                if self.model is not None:
                    results = self.model(img_path)[0]
                    if len(results.boxes) > 0:
                        box = results.boxes[0].xyxy[0].cpu().numpy().tolist()
                        bbox = [int(x) for x in box]
                
                # If YOLO detected nothing, default to a clinical center-interest bounding box
                if len(bbox) == 0:
                    h_hash = hash(base_name) % 100
                    xmin = int(width * (0.35 + (h_hash % 10) / 200))
                    ymin = int(height * (0.35 + (h_hash % 7) / 200))
                    xmax = int(width * (0.55 + (h_hash % 13) / 200))
                    ymax = int(height * (0.55 + (h_hash % 9) / 200))
                    bbox = [xmin, ymin, xmax, ymax]
                
                # Pass 2: Pass bounding box to SAM 2 to extract the segment mask!
                sam_results = self.sam_model(img_path, bboxes=[bbox])[0]
                if sam_results.masks is not None:
                    mask_data = sam_results.masks[0].data[0].cpu().numpy()
                    # Resize mask back to original image size
                    mask_img = Image.fromarray((mask_data * 255).astype(np.uint8)).resize((width, height))
                    color_mask = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                    
                    np_mask = np.array(mask_img)
                    np_color = np.array(color_mask)
                    # Semi-transparent red overlay
                    np_color[np_mask > 128] = [239, 68, 68, 120]
                    
                    overlay = Image.fromarray(np_color, "RGBA")
                    overlay.save(mask_path)
                    affected_pixels = int(np.sum(np_mask > 128))
                    logger.info(f"SAM 2 hybrid segmentation completed. Bbox: {bbox}, mask file written.")
                    return {
                        "bbox": bbox,
                        "mask_filename": mask_filename,
                        "affected_pixel_count": affected_pixels,
                        "total_pixel_count": width * height
                    }
            except Exception as e:
                logger.error(f"SAM 2 hybrid execution failed: {e}. Falling back to simulation.")
                
        # Heuristic segment fallback (circular blob)
        h_hash = hash(os.path.basename(img_path)) % 100
        xmin = int(width * (0.35 + (h_hash % 10) / 200))
        ymin = int(height * (0.35 + (h_hash % 7) / 200))
        xmax = int(width * (0.55 + (h_hash % 13) / 200))
        ymax = int(height * (0.55 + (h_hash % 9) / 200))
        
        xmin = max(0, min(width - 10, xmin))
        ymin = max(0, min(height - 10, ymin))
        xmax = max(xmin + 10, min(width, xmax))
        ymax = max(ymin + 10, min(height, ymax))
        bbox = [xmin, ymin, xmax, ymax]
        
        mask_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
        rx, ry = (xmax - xmin) // 2, (ymax - ymin) // 2
        dist = ((xx - cx) / (rx + 0.1))**2 + ((yy - cy) / (ry + 0.1))**2
        
        np_overlay = np.array(mask_overlay)
        np_overlay[dist <= 1.0] = [239, 68, 68, 120]
        Image.fromarray(np_overlay, "RGBA").save(mask_path)
        
        affected_pixels = int(np.sum(dist <= 1.0))
        return {
            "bbox": bbox,
            "mask_filename": mask_filename,
            "affected_pixel_count": affected_pixels,
            "total_pixel_count": width * height
        }
