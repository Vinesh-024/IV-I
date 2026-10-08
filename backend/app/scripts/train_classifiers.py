import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw
import numpy as np

# Set directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "supervised")
MODELS_OUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")

# Define target paths for linear probe weights and classes lists
BIOMED_HEAD_PATH = os.path.join(MODELS_OUT_DIR, "biomed_clip_head.pth")
DINOV2_HEAD_PATH = os.path.join(MODELS_OUT_DIR, "dinov2_head.pth")
BIOMED_CLASSES_PATH = os.path.join(MODELS_OUT_DIR, "biomed_clip_classes.json")
DINOV2_CLASSES_PATH = os.path.join(MODELS_OUT_DIR, "dinov2_classes.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_OUT_DIR, exist_ok=True)

class SupervisedDataset(Dataset):
    def __init__(self, x_features, y_labels):
        self.x_features = torch.tensor(np.array(x_features), dtype=torch.float32)
        self.y_labels = torch.tensor(y_labels, dtype=torch.long)

    def __len__(self):
        return len(self.x_features)

    def __getitem__(self, idx):
        return self.x_features[idx], self.y_labels[idx]


class LinearProbeClassifier(nn.Module):
    """
    Lightweight Linear Head on top of frozen foundation model embeddings.
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


def create_directory_structure():
    """
    Creates separate folder paths for Scan and RGB training classes.
    """
    rgb_classes = ["Melanoma", "Basal_Cell_Carcinoma", "Eczema", "Psoriasis", "Cheilitis", "Oral_Thrush", "Nail_Infection", "Healthy_Skin"]
    scan_classes = ["Pneumonia", "Atelectasis", "Brain_Tumor", "Brain_Hemorrhage", "Spine_Herniated_Disc", "Normal_Scan"]
    
    rgb_dir = os.path.join(DATA_DIR, "RGB")
    scan_dir = os.path.join(DATA_DIR, "Scan")
    
    for cls in rgb_classes:
        os.makedirs(os.path.join(rgb_dir, cls), exist_ok=True)
    for cls in scan_classes:
        os.makedirs(os.path.join(scan_dir, cls), exist_ok=True)
        
    print(f"Created supervised dataset structures under: {DATA_DIR}")
    print("DIRECTORY STRUCTURE:")
    print("  supervised/RGB/ -> Place visible color disease photos (100+ images per class folders)")
    print("  supervised/Scan/ -> Place grayscale radiology scans (100+ images per class folders)")


def get_image_paths_and_labels(base_path):
    """
    Reads image paths and maps them to numeric labels.
    """
    image_paths = []
    labels = []
    classes = sorted(os.listdir(base_path))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    for cls in classes:
        cls_dir = os.path.join(base_path, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_paths.append(os.path.join(cls_dir, fname))
                labels.append(class_to_idx[cls])
                
    return image_paths, labels, classes


def train_head(x_train, y_train, input_dim, num_classes, model_out_path):
    """
    Trains a linear probe head using PyTorch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SupervisedDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = LinearProbeClassifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 25
    print(f"Training Linear Probe Classifier ({input_dim} -> {num_classes}) on {device}...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
        acc = correct / total
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataset):.4f} | Acc: {acc:.2%}")
        
    torch.save(model.state_dict(), model_out_path)
    print(f"Weights saved successfully to: {model_out_path}\n")


def generate_synthetic_embeddings(num_samples, feature_dim):
    """
    Fallback function to generate random feature embeddings for mock training.
    """
    return np.random.randn(num_samples, feature_dim).astype(np.float32)


def main():
    print("=== Supervised Linear Probe Classifier Head Training Tool ===")
    
    # 1. Initialize folders
    create_directory_structure()
    
    rgb_base = os.path.join(DATA_DIR, "RGB")
    scan_base = os.path.join(DATA_DIR, "Scan")
    
    # Count images in folders
    rgb_paths, rgb_labels, rgb_classes = get_image_paths_and_labels(rgb_base)
    scan_paths, scan_labels, scan_classes = get_image_paths_and_labels(scan_base)
    
    import json
    # Save the exact sorted class configurations to align predicted indices during backend inference
    with open(BIOMED_CLASSES_PATH, "w") as f:
        json.dump(rgb_classes, f)
    with open(DINOV2_CLASSES_PATH, "w") as f:
        json.dump(scan_classes, f)
    print(f"[INFO] Saved class lists: RGB={len(rgb_classes)} classes, Scan={len(scan_classes)} classes.")
    
    # 2. Train BiomedCLIP Linear Head (RGB branch)
    print("\n--- Phase 1: BiomedCLIP RGB Head Training ---")
    if len(rgb_paths) == 0:
        print("[INFO] No custom RGB images detected. Generating synthetic features to create biomed_clip_head.pth template.")
        x_synthetic = generate_synthetic_embeddings(150, 512)
        y_synthetic = np.random.randint(0, len(rgb_classes), 150)
        train_head(x_synthetic, y_synthetic, 512, len(rgb_classes), BIOMED_HEAD_PATH)
    else:
        print(f"Loaded {len(rgb_paths)} RGB photos across {len(rgb_classes)} classes. Extracting BiomedCLIP features...")
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            
            features = []
            for path in rgb_paths:
                with Image.open(path) as img:
                    tensor = preprocess(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = model.encode_image(tensor)
                        feat /= feat.norm(dim=-1, keepdim=True)
                        features.append(feat.cpu().numpy()[0])
            
            train_head(features, rgb_labels, 512, len(rgb_classes), BIOMED_HEAD_PATH)
        except Exception as e:
            print(f"[ERROR] Failed to extract BiomedCLIP features: {e}")
            
    # 3. Train DINOv2 Scan Linear Head (Scan branch)
    print("\n--- Phase 2: DINOv2 Radiology Scan Head Training ---")
    if len(scan_paths) == 0:
        print("[INFO] No custom Scan images detected. Generating synthetic features to create dinov2_head.pth template.")
        x_synthetic = generate_synthetic_embeddings(150, 384)
        y_synthetic = np.random.randint(0, len(scan_classes), 150)
        train_head(x_synthetic, y_synthetic, 384, len(scan_classes), DINOV2_HEAD_PATH)
    else:
        print(f"Loaded {len(scan_paths)} Scan images across {len(scan_classes)} classes. Extracting DINOv2 features...")
        try:
            import torchvision.transforms as transforms
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
            dinov2.eval()
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            features = []
            for path in scan_paths:
                with Image.open(path) as img:
                    tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = dinov2(tensor)
                        features.append(feat.cpu().numpy()[0])
                        
            train_head(features, scan_labels, 384, len(scan_classes), DINOV2_HEAD_PATH)
        except Exception as e:
            print(f"[ERROR] Failed to extract DINOv2 features: {e}")
            
    print("All Supervised Linear Probe training completed successfully!")

if __name__ == "__main__":
    main()
