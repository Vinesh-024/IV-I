import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw
import numpy as np

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPERVISED_DIR = os.path.join(BASE_DIR, "data", "supervised")
MODEL_OUT_PATH = os.path.join(os.path.dirname(BASE_DIR), "models", "modality_model.pth")

os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)

class ModalityDataset(Dataset):
    def __init__(self, x_data, y_labels, transform=None):
        self.x_data = x_data
        self.y_labels = y_labels
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        img_path = self.x_data[idx]
        label = self.y_labels[idx]
        
        try:
            with Image.open(img_path) as img:
                # Convert to RGB and resize to 128x128
                pil_img = img.convert("RGB").resize((128, 128))
                img_np = np.array(pil_img).astype(np.float32) / 255.0
        except Exception as e:
            # Fallback to zero tensor if image fails to load
            img_np = np.zeros((128, 128, 3), dtype=np.float32)

        # HWC to CHW
        img_tensor = torch.tensor(img_np).permute(2, 0, 1)
        return img_tensor, label

class SimpleModalityCNN(nn.Module):
    """
    Lightweight CNN to classify images into Scan (0) or RGB (1).
    """
    def __init__(self):
        super(SimpleModalityCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 128 -> 64
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64 -> 32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 32 -> 16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2) # Outputs: [Scan, RGB]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_real_data():
    """
    Scans the supervised directory for actual RGB and Scan images.
    Returns lists of image file paths and corresponding labels (0=Scan, 1=RGB).
    """
    x_paths = []
    y_labels = []

    # 1. Scan images (0)
    scan_dir = os.path.join(SUPERVISED_DIR, "Scan")
    if os.path.exists(scan_dir):
        for root, _, files in os.walk(scan_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                    x_paths.append(os.path.join(root, file))
                    y_labels.append(0)

    # 2. RGB images (1)
    rgb_dir = os.path.join(SUPERVISED_DIR, "RGB")
    if os.path.exists(rgb_dir):
        for root, _, files in os.walk(rgb_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                    x_paths.append(os.path.join(root, file))
                    y_labels.append(1)

    return x_paths, y_labels

def main():
    print("=== Supervised Modality Router Trainer ===")
    
    # 1. Prepare data
    file_paths, labels = load_real_data()
    num_samples = len(file_paths)
    
    if num_samples < 20:
        print(f"Warning: Only found {num_samples} real images. Modality trainer requires at least 20 images.")
        return
        
    print(f"Found {num_samples} real clinical images for modality training:")
    print(f"  - Scans: {labels.count(0)}")
    print(f"  - RGB photos: {labels.count(1)}")

    # Shuffle and split into train/val datasets
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    split_idx = int(0.85 * num_samples)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    x_train = [file_paths[i] for i in train_indices]
    y_train = [labels[i] for i in train_indices]
    x_val = [file_paths[i] for i in val_indices]
    y_val = [labels[i] for i in val_indices]

    train_dataset = ModalityDataset(x_train, y_train)
    val_dataset = ModalityDataset(x_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 2. Setup Model & training hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = SimpleModalityCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Train loop
    epochs = 8
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
        train_loss /= len(train_loader.dataset)
        train_acc = correct / total
        
        # Validation pass
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
        
    # 4. Save model
    print(f"Saving trained model weights to: {MODEL_OUT_PATH}")
    torch.save(model.state_dict(), MODEL_OUT_PATH)
    print("Modality Router training completed successfully!")

if __name__ == "__main__":
    main()
