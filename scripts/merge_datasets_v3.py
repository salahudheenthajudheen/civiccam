"""
CivicCam Dataset Merger v3
Merges all datasets:
1. License plate dataset
2. Waste disposal dataset  
3. New auto-labelled images (655 images)

Classes:
- license_plate (index 0)
- object (index 1)
- public (index 2)
- waste (index 3)
"""

import os
import shutil
import yaml
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"

# Source datasets
LICENSE_PLATE_DIR = DATASETS_DIR / "license_plate"
WASTE_DIR = DATASETS_DIR / "waste_disposal_v2"
NEW_IMAGES_DIR = DATASETS_DIR / "new_images"

# Output
COMBINED_DIR = DATASETS_DIR / "combined_v3"
TARGET_CLASSES = ["license_plate", "object", "public", "waste"]


def get_classes_from_yaml(yaml_path):
    """Extract class names from data.yaml"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', [])


def count_files(directory, ext="*.jpg"):
    """Count files in directory"""
    return sum(1 for _ in Path(directory).rglob(ext))


def create_combined_dataset():
    """Create merged dataset from all sources"""
    print("=" * 60)
    print("CivicCam Dataset Merger v3")
    print("=" * 60)
    
    # Clean up old combined directory
    if COMBINED_DIR.exists():
        print(f"\nRemoving existing {COMBINED_DIR.name}...")
        shutil.rmtree(COMBINED_DIR)
    
    # Create directories
    for split in ["train", "valid", "test"]:
        (COMBINED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (COMBINED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    
    stats = {
        "license_plate": {"train": 0, "valid": 0, "test": 0},
        "waste_disposal": {"train": 0, "valid": 0, "test": 0},
        "new_images": {"train": 0, "valid": 0, "test": 0}
    }
    
    # ===== 1. Process License Plate Dataset =====
    print("\n[1/3] Processing License Plate Dataset...")
    
    for split in ["train", "valid", "test"]:
        images_dir = LICENSE_PLATE_DIR / split / "images"
        labels_dir = LICENSE_PLATE_DIR / split / "labels"
        
        if not images_dir.exists():
            print(f"  Skipping {split} - not found")
            continue
        
        dest_images = COMBINED_DIR / split / "images"
        dest_labels = COMBINED_DIR / split / "labels"
        
        # Copy images and remap labels
        for img_file in images_dir.glob("*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                new_name = f"lp_{img_file.name}"
                shutil.copy2(img_file, dest_images / new_name)
                stats["license_plate"][split] += 1
        
        for label_file in labels_dir.glob("*.txt"):
            new_name = f"lp_{label_file.name}"
            
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # All classes -> 0 (license_plate)
            remapped_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    parts[0] = "0"
                    remapped_lines.append(" ".join(parts) + "\n")
            
            with open(dest_labels / new_name, 'w') as f:
                f.writelines(remapped_lines)
    
    lp_total = sum(stats["license_plate"].values())
    print(f"  ✓ License Plate: {lp_total} images")
    
    # ===== 2. Process Waste Disposal Dataset =====
    print("\n[2/3] Processing Waste Disposal Dataset...")
    
    # Get class mapping
    waste_yaml = WASTE_DIR / "data.yaml"
    if waste_yaml.exists():
        waste_classes = get_classes_from_yaml(waste_yaml)
        
        # Mapping for waste dataset classes
        manual_mapping = {
            "litter": "waste",
            "waste": "waste",
            "object": "object",
            "other-unknown": "object",
            "public": "public"
        }
        
        waste_class_map = {}
        for old_idx, class_name in enumerate(waste_classes):
            target_name = manual_mapping.get(class_name, class_name)
            if target_name in TARGET_CLASSES:
                waste_class_map[old_idx] = TARGET_CLASSES.index(target_name)
        
        for split in ["train", "valid", "test"]:
            images_dir = WASTE_DIR / split / "images"
            labels_dir = WASTE_DIR / split / "labels"
            
            if not images_dir.exists():
                continue
            
            dest_images = COMBINED_DIR / split / "images"
            dest_labels = COMBINED_DIR / split / "labels"
            
            for img_file in images_dir.glob("*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    new_name = f"wd_{img_file.name}"
                    shutil.copy2(img_file, dest_images / new_name)
                    stats["waste_disposal"][split] += 1
            
            for label_file in labels_dir.glob("*.txt"):
                new_name = f"wd_{label_file.name}"
                
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                remapped_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_class = int(parts[0])
                        new_class = waste_class_map.get(old_class, old_class)
                        parts[0] = str(new_class)
                        remapped_lines.append(" ".join(parts) + "\n")
                
                with open(dest_labels / new_name, 'w') as f:
                    f.writelines(remapped_lines)
    
    wd_total = sum(stats["waste_disposal"].values())
    print(f"  ✓ Waste Disposal: {wd_total} images")
    
    # ===== 3. Process New Auto-Labelled Images =====
    print("\n[3/3] Processing New Auto-Labelled Images...")
    
    if NEW_IMAGES_DIR.exists():
        for split in ["train", "valid", "test"]:
            images_dir = NEW_IMAGES_DIR / split / "images"
            labels_dir = NEW_IMAGES_DIR / split / "labels"
            
            if not images_dir.exists():
                continue
            
            dest_images = COMBINED_DIR / split / "images"
            dest_labels = COMBINED_DIR / split / "labels"
            
            # Copy images (no renaming needed, already prefixed)
            for img_file in images_dir.glob("*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    shutil.copy2(img_file, dest_images / img_file.name)
                    stats["new_images"][split] += 1
            
            # Copy labels (already in correct format)
            for label_file in labels_dir.glob("*.txt"):
                shutil.copy2(label_file, dest_labels / label_file.name)
    
    new_total = sum(stats["new_images"].values())
    print(f"  ✓ New Images: {new_total} images")
    
    # ===== Create data.yaml =====
    data_yaml = {
        "path": str(COMBINED_DIR.absolute()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(TARGET_CLASSES),
        "names": TARGET_CLASSES
    }
    
    yaml_path = COMBINED_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    # ===== Summary =====
    print("\n" + "=" * 60)
    print("Dataset Merge Complete!")
    print("=" * 60)
    
    print(f"\nOutput: {COMBINED_DIR}")
    print(f"\nClasses ({len(TARGET_CLASSES)}):")
    for i, cls in enumerate(TARGET_CLASSES):
        print(f"  {i}: {cls}")
    
    total_train = stats["license_plate"]["train"] + stats["waste_disposal"]["train"] + stats["new_images"]["train"]
    total_valid = stats["license_plate"]["valid"] + stats["waste_disposal"]["valid"] + stats["new_images"]["valid"]
    total_test = stats["license_plate"]["test"] + stats["waste_disposal"]["test"] + stats["new_images"]["test"]
    
    print(f"\nDataset Statistics:")
    print(f"  {'Dataset':<20} {'Train':>8} {'Valid':>8} {'Test':>8} {'Total':>8}")
    print(f"  {'-' * 52}")
    print(f"  {'License Plate':<20} {stats['license_plate']['train']:>8} {stats['license_plate']['valid']:>8} {stats['license_plate']['test']:>8} {lp_total:>8}")
    print(f"  {'Waste Disposal':<20} {stats['waste_disposal']['train']:>8} {stats['waste_disposal']['valid']:>8} {stats['waste_disposal']['test']:>8} {wd_total:>8}")
    print(f"  {'New Images':<20} {stats['new_images']['train']:>8} {stats['new_images']['valid']:>8} {stats['new_images']['test']:>8} {new_total:>8}")
    print(f"  {'-' * 52}")
    print(f"  {'TOTAL':<20} {total_train:>8} {total_valid:>8} {total_test:>8} {lp_total + wd_total + new_total:>8}")
    
    print(f"\n✅ Ready for training with: python scripts/train_model.py")
    
    return yaml_path


if __name__ == "__main__":
    create_combined_dataset()
