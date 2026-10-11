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
    rgb_classes = [
        "Melanoma", "Basal_Cell_Carcinoma", "Eczema", "Psoriasis", 
        "Cheilitis", "Oral_Thrush", 
        # "Nail_Infection",  # Temporarily kept aside (empty folder / 0 images; covered by Onychomycosis & Nail_Psoriasis)
        "Healthy_Skin"
    ]
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
    Filters out temporarily discarded classes and empty folders.
    """
    # Classes temporarily discarded / kept aside (preserved in codebase, but skipped during training)
    TEMPORARILY_DISCARDED_CLASSES = {
        "Nail_Infection",  # Kept aside: empty folder (0 images); specific nail conditions handled by Onychomycosis & Nail_Psoriasis
    }

    image_paths = []
    labels = []
    
    candidate_classes = sorted([
        cls for cls in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, cls))
        and cls not in TEMPORARILY_DISCARDED_CLASSES
        and not cls.startswith(('_', '.'))
    ])
    
    # Filter out any directory that has 0 valid image files
    classes = []
    for cls in candidate_classes:
        cls_dir = os.path.join(base_path, cls)
        has_imgs = any(fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) for fname in os.listdir(cls_dir))
        if has_imgs:
            classes.append(cls)
        else:
            print(f"  [NOTE] Skipping empty or discarded folder: '{cls}' (0 valid images)")

    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    for cls in classes:
        cls_dir = os.path.join(base_path, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_paths.append(os.path.join(cls_dir, fname))
                labels.append(class_to_idx[cls])
                
    return image_paths, labels, classes


def train_head(x_train, y_train, input_dim, num_classes, model_out_path, val_split=0.2, seed=42):
    """
    Trains a linear probe head with a stratified train/val split.
    Tracks both training and validation accuracy per epoch.
    Saves best weights based on validation accuracy (not training accuracy).
    Applies early stopping to prevent overfitting.
    """
    import json, random
    from collections import defaultdict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Stratified 80/20 split ──────────────────────────────────────────────
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for i, label in enumerate(y_train):
        by_class[label].append(i)

    train_idx, val_idx = [], []
    for cls_indices in by_class.values():
        rng.shuffle(cls_indices)
        n_val = max(1, int(len(cls_indices) * val_split))
        val_idx.extend(cls_indices[:n_val])
        train_idx.extend(cls_indices[n_val:])

    x_arr = np.array(x_train)
    y_arr = np.array(y_train)

    train_dataset = SupervisedDataset(x_arr[train_idx], y_arr[train_idx])
    val_dataset   = SupervisedDataset(x_arr[val_idx],   y_arr[val_idx])
    train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=16, shuffle=False)

    print(f"  Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples | Classes: {num_classes}")

    model = LinearProbeClassifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epochs        = 40   # More epochs — early stopping will cut it short if needed
    patience      = 7    # Early stopping patience
    best_val_acc  = 0.0
    no_improve    = 0
    history       = []

    print(f"Training LinearProbe ({input_dim} → {num_classes}) on {device}")
    print(f"  {'Epoch':<8} {'Train Loss':<14} {'Train Acc':<14} {'Val Acc':<12} {'Status'}")
    print(f"  {'─'*60}")

    for epoch in range(epochs):
        # ── Training pass ──────────────────────────────────────
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out  = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * batch_x.size(0)
            _, pred    = out.max(1)
            t_total   += batch_y.size(0)
            t_correct += pred.eq(batch_y).sum().item()

        scheduler.step()
        train_acc  = t_correct / t_total
        train_loss = t_loss / len(train_dataset)

        # ── Validation pass ────────────────────────────────────
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out = model(batch_x)
                _, pred = out.max(1)
                v_total   += batch_y.size(0)
                v_correct += pred.eq(batch_y).sum().item()
        val_acc = v_correct / v_total

        # ── Best model checkpoint ──────────────────────────────
        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_out_path)
            no_improve = 0
            status = "✓ saved"
        else:
            no_improve += 1
            gap = train_acc - val_acc
            if gap > 0.15:
                status = f"⚠ overfit gap={gap:.1%}"
            elif train_acc < 0.55:
                status = "↓ underfit"

        history.append({
            "epoch":     epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc, 4),
            "val_acc":    round(val_acc, 4),
            "gap":        round(train_acc - val_acc, 4)
        })

        print(f"  {epoch+1:<8} {train_loss:<14.4f} {train_acc:<14.2%} {val_acc:<12.2%} {status}")

        # ── Early stopping ─────────────────────────────────────
        if no_improve >= patience:
            print(f"  Early stopping triggered — no val improvement for {patience} epochs.")
            break

    print(f"\n  Best Val Accuracy : {best_val_acc:.2%}")
    print(f"  Final Train Acc   : {train_acc:.2%}")
    gap = train_acc - best_val_acc
    if gap > 0.15:
        print(f"  ⚠  OVERFITTING DETECTED: train-val gap = {gap:.2%}")
    elif best_val_acc < 0.55:
        print(f"  ↓  UNDERFITTING DETECTED: val accuracy too low ({best_val_acc:.2%})")
    else:
        print(f"  ✓  Model appears well-fitted (gap = {gap:.2%})")

    # Save training history JSON alongside the weights
    history_path = model_out_path.replace(".pth", "_training_history.json")
    with open(history_path, "w") as f:
        json.dump({"best_val_acc": round(best_val_acc, 4), "epochs_run": len(history), "history": history}, f, indent=2)
    print(f"  Training history saved: {history_path}\n")
    return best_val_acc


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
