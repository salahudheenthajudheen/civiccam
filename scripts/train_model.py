"""
CivicCam Model Training Script
Trains YOLOv8 for littering detection with 4 classes:
- license_plate: Vehicle number plates
- object: Objects/items being thrown
- public: Public spaces/areas
- waste: Litter/waste on ground
"""

from ultralytics import YOLO
from pathlib import Path
import yaml

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATASET_PATH = BASE_DIR / "datasets" / "combined_v2" / "data.yaml"
OUTPUT_DIR = BASE_DIR / "runs"
MODELS_DIR = BASE_DIR / "models"

# Training parameters
TRAINING_CONFIG = {
    "model": "yolov8n.pt",     # Nano model for speed detection
    "epochs": 50,              # Number of epochs
    "batch": 16,               # Batch size
    "imgsz": 640,              # Image size
    "patience": 15,            # Early stopping patience
    "device": 0,               # GPU device (0) or cpu
    "project": str(BASE_DIR / "runs" / "detect"),
    "name": "civiccam_v2",     # Project name
    "exist_ok": True,          # Overwrite existing project
    "pretrained": True,        # Use pretrained weights
    "optimizer": "auto",       # Optimizer
    "verbose": True,           # Verbose output
    "seed": 42,                # Random seed for reproducibility
    "plots": True,             # Save plots
    "save": True,              # Save checkpoints
    
    # Missing keys restored
    "lr0": 0.01,
    "lrf": 0.01,
    
    # Augmentation
    "augment": True,
    "mosaic": 1.0,
    "mixup": 0.1,
    "copy_paste": 0.1,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.5,
    "flipud": 0.0,
    "fliplr": 0.5,
    
    # Performance
    "workers": 4,
    "cache": True,
}


def train_model():
    """Train YOLOv8 model for CivicCam"""
    print("=" * 60)
    print("CivicCam Model Training")
    print("=" * 60)
    
    # Load dataset config
    print(f"\nDataset: {DATASET_PATH}")
    with open(DATASET_PATH, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"Classes: {data_config['names']}")
    print(f"Number of classes: {data_config['nc']}")
    
    # Check device
    import torch
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    print(f"Training device: {device}")
    
    if not torch.cuda.is_available():
        print("\n[!] Warning: Training on CPU will be slow!")
        print("    For faster training, use Google Colab with GPU.")
    
    # Load pretrained model
    print(f"\nLoading model: {TRAINING_CONFIG['model']}")
    model = YOLO(TRAINING_CONFIG["model"])
    
    # Start training
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    
    results = model.train(
        data=str(DATASET_PATH),
        epochs=TRAINING_CONFIG["epochs"],
        batch=TRAINING_CONFIG["batch"],
        imgsz=TRAINING_CONFIG["imgsz"],
        patience=TRAINING_CONFIG["patience"],
        optimizer=TRAINING_CONFIG["optimizer"],
        lr0=TRAINING_CONFIG["lr0"],
        lrf=TRAINING_CONFIG["lrf"],
        augment=TRAINING_CONFIG["augment"],
        mosaic=TRAINING_CONFIG["mosaic"],
        mixup=TRAINING_CONFIG["mixup"],
        copy_paste=TRAINING_CONFIG["copy_paste"],
        degrees=TRAINING_CONFIG["degrees"],
        translate=TRAINING_CONFIG["translate"],
        scale=TRAINING_CONFIG["scale"],
        flipud=TRAINING_CONFIG["flipud"],
        fliplr=TRAINING_CONFIG["fliplr"],
        workers=TRAINING_CONFIG["workers"],
        cache=TRAINING_CONFIG["cache"],
        project=TRAINING_CONFIG["project"],
        name=TRAINING_CONFIG["name"],
        exist_ok=TRAINING_CONFIG["exist_ok"],
        device=TRAINING_CONFIG["device"],
        verbose=True,
        plots=True,
    )
    
    # Training complete
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Get best model path
    best_model = Path(TRAINING_CONFIG["project"]) / TRAINING_CONFIG["name"] / "weights" / "best.pt"
    print(f"\nBest model saved to: {best_model}")
    
    # Copy best model to models directory
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    
    import shutil
    final_model_path = models_dir / "civiccam_best.pt"
    shutil.copy(best_model, final_model_path)
    print(f"Model copied to: {final_model_path}")
    
    # Validate model
    print("\n" + "=" * 60)
    print("Validating Model...")
    print("=" * 60)
    
    metrics = model.val()
    
    print(f"\nValidation Results:")
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    
    return final_model_path


if __name__ == "__main__":
    train_model()
