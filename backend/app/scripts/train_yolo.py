import os
import yaml
from ultralytics import YOLO

# Get absolute path for output configs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data", "yolo_dataset")
YAML_OUT_PATH = os.path.join(DATASET_DIR, "dataset.yaml")

def setup_yolo_folders():
    """
    Creates standard YOLO directory structures for annotated training images.
    """
    dirs = [
        "images/train", "images/val",
        "labels/train", "labels/val"
    ]
    for d in dirs:
        os.makedirs(os.path.join(DATASET_DIR, d), exist_ok=True)
    print(f"Created YOLO folder structure under: {DATASET_DIR}")
    print("INSTRUCTIONS:")
    print("1. Place your training images inside 'images/train/' and validation images inside 'images/val/'")
    print("2. Place your txt label polygon files inside 'labels/train/' and 'labels/val/'")
    print("   Each label file must match the name of the image (e.g. img1.jpg -> img1.txt)")
    print("   Txt format: class_id x1 y1 x2 y2 x3 y3 ... (normalized polygon boundary points [0..1])")

def write_yolo_yaml():
    """
    Writes YOLO dataset config yaml file.
    """
    # Map classes based on visible diseases and scan pathologies
    classes = [
        "Melanoma", "Basal_Cell_Carcinoma", "Eczema", "Psoriasis", 
        "Conjunctivitis", "Stye", "Blepharitis", "Paronychia",
        "Finger_Fracture", "Cut_Laceration", "Onychomycosis",
        "Diabetic_Ulcer", "Pressure_Ulcer", "Surgical_Incision",
        "Burn_Wound", "Pneumonia", "Atelectasis", "Pleural_Effusion",
        "Brain_Tumor", "Brain_Hemorrhage", "Spine_Herniated_Disc"
    ]
    
    yaml_data = {
        "path": os.path.abspath(DATASET_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(classes)}
    }
    
    with open(YAML_OUT_PATH, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    print(f"Generated YOLO dataset YAML config at: {YAML_OUT_PATH}")

def main():
    print("=== YOLOv11-Seg Custom Model Training Utility ===")
    
    # 1. Setup folders and config
    setup_yolo_folders()
    write_yolo_yaml()
    
    # 2. Check if datasets are seeded. If empty, print template training guidelines
    train_images = os.listdir(os.path.join(DATASET_DIR, "images", "train"))
    
    if len(train_images) == 0:
        print("\n[WARNING] No images detected in 'images/train'. Seeding is required before training.")
        print("Once your dataset is populated, you can execute training using: model.train(...)")
        print("\nSAMPLE PYTHON TRAINING SCRIPT TEMPLATE:")
        print("-" * 50)
        print("""from ultralytics import YOLO

# Load a pre-trained YOLOv11 Nano Segmentation model
model = YOLO("yolov11n-seg.pt")

# Train the model on your custom dataset
results = model.train(
    data="{}", 
    epochs=50, 
    imgsz=640, 
    batch=16,
    device="cuda",  # uses GPU
    workers=4
)
""".format(os.path.abspath(YAML_OUT_PATH)))
        print("-" * 50)
    else:
        # Run local training
        print(f"\nSeeded dataset detected with {len(train_images)} training images. Starting YOLOv11 training...")
        try:
            model = YOLO("yolov11n-seg.pt")
            model.train(data=YAML_OUT_PATH, epochs=50, imgsz=640, device="cuda")
            print("YOLOv11-Seg training completed!")
        except Exception as e:
            print(f"Error executing YOLO training: {e}")

if __name__ == "__main__":
    main()
