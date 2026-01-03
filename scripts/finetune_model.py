"""
CivicCam Model Fine-Tuning Script
Fine-tunes from existing trained model for improved performance
"""

from ultralytics import YOLO
from pathlib import Path
import yaml
import torch

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATASET_PATH = BASE_DIR / "datasets" / "combined_v2" / "data.yaml"
OUTPUT_DIR = BASE_DIR / "runs"
MODELS_DIR = BASE_DIR / "models"

# Existing trained model to fine-tune from
PRETRAINED_MODEL = MODELS_DIR / "civiccam_best.pt"

# Fine-tuning optimized parameters
FINETUNE_CONFIG = {
    # Lower learning rate for fine-tuning (prevents catastrophic forgetting)
    "lr0": 0.001,                  # Initial LR (10x lower than training)
    "lrf": 0.01,                   # Final LR factor
    "cos_lr": True,                # Cosine LR scheduler
    "warmup_epochs": 3.0,          # Warmup epochs
    "warmup_momentum": 0.8,        # Warmup momentum
    
    # Training settings
    "epochs": 30,                  # Fine-tuning epochs
    "batch": 16,                   # Batch size
    "imgsz": 640,                  # Image size
    "patience": 10,                # Early stopping patience
    "device": 0,                   # GPU device
    
    # Project settings
    "project": str(OUTPUT_DIR / "detect"),
    "name": "civiccam_finetuned",
    "exist_ok": True,
    
    # Augmentation (lighter for fine-tuning)
    "augment": True,
    "mosaic": 0.5,                 # Reduced mosaic
    "mixup": 0.05,                 # Reduced mixup
    "copy_paste": 0.0,             # Disabled copy-paste
    "degrees": 5.0,                # Less rotation
    "translate": 0.1,
    "scale": 0.3,                  # Less scale variation
    "flipud": 0.0,
    "fliplr": 0.5,
    
    # Performance
    "workers": 4,
    "cache": True,
    "verbose": True,
    "plots": True,
    "save": True,
}


def finetune_model():
    """Fine-tune CivicCam model from existing checkpoint"""
    print("=" * 60)
    print("CivicCam Model Fine-Tuning")
    print("=" * 60)
    
    # Check for existing model
    if not PRETRAINED_MODEL.exists():
        print(f"\n[!] Pretrained model not found: {PRETRAINED_MODEL}")
        print("    Run train_model.py first or download the model.")
        return None
    
    # Load dataset config
    print(f"\nDataset: {DATASET_PATH}")
    with open(DATASET_PATH, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"Classes: {data_config['names']}")
    print(f"Number of classes: {data_config['nc']}")
    
    # Check device
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    print(f"Training device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("\n[!] Warning: Fine-tuning on CPU will be slow!")
        FINETUNE_CONFIG["device"] = "cpu"
    
    # Load pretrained model
    print(f"\nLoading model: {PRETRAINED_MODEL}")
    model = YOLO(str(PRETRAINED_MODEL))
    
    # Start fine-tuning
    print("\n" + "=" * 60)
    print("Starting Fine-Tuning...")
    print("=" * 60)
    print(f"\nKey parameters:")
    print(f"  Learning rate: {FINETUNE_CONFIG['lr0']}")
    print(f"  Epochs: {FINETUNE_CONFIG['epochs']}")
    print(f"  Batch size: {FINETUNE_CONFIG['batch']}")
    print(f"  Cosine LR: {FINETUNE_CONFIG['cos_lr']}")
    print(f"  Warmup: {FINETUNE_CONFIG['warmup_epochs']} epochs")
    
    results = model.train(
        data=str(DATASET_PATH),
        epochs=FINETUNE_CONFIG["epochs"],
        batch=FINETUNE_CONFIG["batch"],
        imgsz=FINETUNE_CONFIG["imgsz"],
        patience=FINETUNE_CONFIG["patience"],
        lr0=FINETUNE_CONFIG["lr0"],
        lrf=FINETUNE_CONFIG["lrf"],
        cos_lr=FINETUNE_CONFIG["cos_lr"],
        warmup_epochs=FINETUNE_CONFIG["warmup_epochs"],
        warmup_momentum=FINETUNE_CONFIG["warmup_momentum"],
        augment=FINETUNE_CONFIG["augment"],
        mosaic=FINETUNE_CONFIG["mosaic"],
        mixup=FINETUNE_CONFIG["mixup"],
        copy_paste=FINETUNE_CONFIG["copy_paste"],
        degrees=FINETUNE_CONFIG["degrees"],
        translate=FINETUNE_CONFIG["translate"],
        scale=FINETUNE_CONFIG["scale"],
        flipud=FINETUNE_CONFIG["flipud"],
        fliplr=FINETUNE_CONFIG["fliplr"],
        workers=FINETUNE_CONFIG["workers"],
        cache=FINETUNE_CONFIG["cache"],
        project=FINETUNE_CONFIG["project"],
        name=FINETUNE_CONFIG["name"],
        exist_ok=FINETUNE_CONFIG["exist_ok"],
        device=FINETUNE_CONFIG["device"],
        verbose=True,
        plots=True,
    )
    
    # Training complete
    print("\n" + "=" * 60)
    print("Fine-Tuning Complete!")
    print("=" * 60)
    
    # Get best model path
    best_model = Path(FINETUNE_CONFIG["project"]) / FINETUNE_CONFIG["name"] / "weights" / "best.pt"
    print(f"\nBest model saved to: {best_model}")
    
    # Copy best model to models directory
    import shutil
    final_model_path = MODELS_DIR / "civiccam_finetuned.pt"
    shutil.copy(best_model, final_model_path)
    print(f"Model copied to: {final_model_path}")
    
    # Validate model
    print("\n" + "=" * 60)
    print("Validating Fine-Tuned Model...")
    print("=" * 60)
    
    metrics = model.val()
    
    print(f"\nValidation Results:")
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    
    # Tips for next steps
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Compare mAP50 with previous model")
    print("2. Test detection: python detect.py --source <image/video>")
    print("3. Update config.py MODEL_PATH to use finetuned model")
    
    return final_model_path


if __name__ == "__main__":
    finetune_model()
