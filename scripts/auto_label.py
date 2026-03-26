"""
CivicCam Auto-Labelling Script
Generates YOLO-format labels for unlabelled images using a trained model.

This uses pseudo-labelling (self-training) approach:
1. Load the existing trained model
2. Run inference on new images
3. Generate label files from confident predictions
4. Split into train/valid/test sets
"""

import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

# Configuration
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "civiccam_best.pt"
SOURCE_DIR = Path("/Users/ar/Documents/new dataset")
OUTPUT_DIR = BASE_DIR / "datasets" / "new_images"

# Labelling parameters
CONFIDENCE_THRESHOLD = 0.35  # Minimum confidence for a detection to be labelled
TRAIN_RATIO = 0.80
VALID_RATIO = 0.15
TEST_RATIO = 0.05

# Classes (must match the model's classes)
CLASSES = ["license_plate", "object", "public", "waste"]


def setup_output_dirs():
    """Create output directory structure"""
    print("\n[1/5] Setting up directory structure...")
    
    # Clean up existing directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    
    # Create train/valid/test directories
    for split in ["train", "valid", "test"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    
    print(f"  Output directory: {OUTPUT_DIR}")


def collect_images():
    """Collect all images from source directory"""
    print("\n[2/5] Collecting images...")
    
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]:
        images.extend(SOURCE_DIR.rglob(ext))
    
    print(f"  Found {len(images)} images")
    return sorted(images)


def split_images(images):
    """Split images into train/valid/test sets"""
    print("\n[3/5] Splitting dataset...")
    
    random.seed(42)  # For reproducibility
    random.shuffle(images)
    
    n = len(images)
    train_end = int(n * TRAIN_RATIO)
    valid_end = train_end + int(n * VALID_RATIO)
    
    splits = {
        "train": images[:train_end],
        "valid": images[train_end:valid_end],
        "test": images[valid_end:]
    }
    
    for split, imgs in splits.items():
        print(f"  {split}: {len(imgs)} images")
    
    return splits


def auto_label_images(model, splits):
    """Run inference and generate YOLO labels"""
    print("\n[4/5] Auto-labelling images...")
    
    stats = {
        "total_images": 0,
        "images_with_detections": 0,
        "total_detections": 0,
        "detections_per_class": {cls: 0 for cls in CLASSES}
    }
    
    for split, images in splits.items():
        print(f"\n  Processing {split} set ({len(images)} images)...")
        
        for i, img_path in enumerate(images):
            # Run inference
            results = model(str(img_path), verbose=False)[0]
            
            # Get image dimensions for YOLO normalization
            img_h, img_w = results.orig_shape
            
            # Generate label content
            label_lines = []
            
            for box in results.boxes:
                conf = float(box.conf[0])
                
                if conf >= CONFIDENCE_THRESHOLD:
                    cls_id = int(box.cls[0])
                    
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Convert to YOLO format (normalized xywh)
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    
                    label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    
                    stats["total_detections"] += 1
                    stats["detections_per_class"][CLASSES[cls_id]] += 1
            
            # Create new filename (to avoid conflicts)
            new_name = f"new_{img_path.parent.name}_{img_path.name}"
            
            # Copy image
            dest_img = OUTPUT_DIR / split / "images" / new_name
            shutil.copy2(img_path, dest_img)
            
            # Write label file
            label_name = Path(new_name).stem + ".txt"
            dest_label = OUTPUT_DIR / split / "labels" / label_name
            
            with open(dest_label, 'w') as f:
                f.write("\n".join(label_lines))
            
            stats["total_images"] += 1
            if label_lines:
                stats["images_with_detections"] += 1
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(images)} images")
        
        print(f"    ✓ {split} complete")
    
    return stats


def create_data_yaml():
    """Create data.yaml for the new dataset"""
    import yaml
    
    data_config = {
        "path": str(OUTPUT_DIR.absolute()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASSES),
        "names": CLASSES
    }
    
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    return yaml_path


def main():
    """Main auto-labelling pipeline"""
    print("=" * 60)
    print("CivicCam Auto-Labelling Script")
    print("=" * 60)
    print(f"\nSource: {SOURCE_DIR}")
    print(f"Model: {MODEL_PATH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    # Check paths
    if not SOURCE_DIR.exists():
        print(f"\n❌ Error: Source directory not found: {SOURCE_DIR}")
        return
    
    if not MODEL_PATH.exists():
        print(f"\n❌ Error: Model not found: {MODEL_PATH}")
        return
    
    # Setup
    setup_output_dirs()
    
    # Collect images
    images = collect_images()
    if not images:
        print("\n❌ No images found!")
        return
    
    # Split dataset
    splits = split_images(images)
    
    # Load model
    print("\n  Loading model...")
    model = YOLO(str(MODEL_PATH))
    print(f"  Model loaded: {len(CLASSES)} classes")
    
    # Auto-label
    stats = auto_label_images(model, splits)
    
    # Create data.yaml
    yaml_path = create_data_yaml()
    
    # Summary
    print("\n" + "=" * 60)
    print("Auto-Labelling Complete!")
    print("=" * 60)
    print(f"\nStatistics:")
    print(f"  Total images processed: {stats['total_images']}")
    print(f"  Images with detections: {stats['images_with_detections']}")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"\nDetections per class:")
    for cls, count in stats['detections_per_class'].items():
        print(f"  {cls}: {count}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Data YAML: {yaml_path}")
    print("\n✅ Ready for merging with existing dataset!")


if __name__ == "__main__":
    main()
