"""
CivicCam Dataset Merger - Corrected Version
Properly merges license plate and waste disposal datasets with unified classes:
- license_plate (from all 725 plate number classes)
- object (from waste dataset)  
- public (from waste dataset)
- waste (from waste dataset)
"""

import os
import shutil
import yaml
from pathlib import Path

# Configuration
DATASETS_DIR = Path("datasets")
LICENSE_PLATE_DIR = DATASETS_DIR / "license_plate"
WASTE_DIR = DATASETS_DIR / "waste_disposal_v2"
COMBINED_DIR = DATASETS_DIR / "combined_v2"
TARGET_CLASSES = ["license_plate", "object", "public", "waste"]


def get_classes_from_yaml(yaml_path):
    """Extract class names from data.yaml"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', [])


def create_combined_dataset():
    """Create properly merged dataset with unified classes"""
    print("=" * 60)
    print("CivicCam Dataset Merger - Corrected Version")
    print("=" * 60)
    
    combined_dir = COMBINED_DIR
    
    # Clean up old combined directory
    if combined_dir.exists():
        shutil.rmtree(combined_dir)
    
    # Create directories
    for split in ["train", "valid", "test"]:
        (combined_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (combined_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    file_count = {"train": 0, "valid": 0, "test": 0}
    
    # Process license plate dataset
    # All classes map to index 0 (license_plate)
    print("\n[1/2] Processing License Plate Dataset...")
    lp_dir = DATASETS_DIR / "license_plate"
    
    for split in ["train", "valid", "test"]:
        images_dir = lp_dir / split / "images"
        labels_dir = lp_dir / split / "labels"
        
        if not images_dir.exists():
            print(f"  Skipping {split} - not found")
            continue
        
        dest_images = combined_dir / split / "images"
        dest_labels = combined_dir / split / "labels"
        
        # Copy images
        for img_file in images_dir.glob("*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                new_name = f"lp_{img_file.name}"
                shutil.copy2(img_file, dest_images / new_name)
                file_count[split] += 1
        
        # Copy and remap labels (all classes -> 0 for license_plate)
        for label_file in labels_dir.glob("*.txt"):
            new_name = f"lp_{label_file.name}"
            
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            remapped_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # All classes become 0 (license_plate)
                    parts[0] = "0"
                    remapped_lines.append(" ".join(parts) + "\n")
            
            with open(dest_labels / new_name, 'w') as f:
                f.writelines(remapped_lines)
        
        print(f"  {split}: {file_count[split]} images processed")
    
    # Reset count for waste dataset
    waste_count = {"train": 0, "valid": 0, "test": 0}
    
    # Process waste disposal dataset
    waste_dir = WASTE_DIR
    
    # Get waste dataset classes
    waste_yaml = waste_dir / "data.yaml"
    print(f"  Reading YAML from {waste_yaml.absolute()}")
    waste_classes = get_classes_from_yaml(waste_yaml)
    print(f"  Waste classes: {waste_classes}")
    
    # Create mapping: old_class -> new_class
    # Target: ["license_plate", "object", "public", "waste"] (indices 0, 1, 2, 3)
    waste_class_map = {}
    
    # Explicit mapping for known classes
    manual_mapping = {
        "litter": "waste",
        "waste": "waste",
        "object": "object",
        "other-unknown": "object",
        "public": "public"
    }
    
    for old_idx, class_name in enumerate(waste_classes):
        target_name = manual_mapping.get(class_name, class_name)
        
        if target_name in TARGET_CLASSES:
            new_idx = TARGET_CLASSES.index(target_name)
            waste_class_map[old_idx] = new_idx
            print(f"  Map {class_name}({old_idx}) -> {target_name}({new_idx})")
        else:
            print(f"  Warning: Skipping class '{class_name}' - not in target list")
            
    print(f"  Final Class mapping: {waste_class_map}")
    
    for split in ["train", "valid", "test"]:
        images_dir = waste_dir / split / "images"
        labels_dir = waste_dir / split / "labels"
        
        if not images_dir.exists():
            print(f"  Skipping {split} - not found")
            continue
        
        dest_images = combined_dir / split / "images"
        dest_labels = combined_dir / split / "labels"
        
        # Copy images
        for img_file in images_dir.glob("*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                new_name = f"wd_{img_file.name}"
                shutil.copy2(img_file, dest_images / new_name)
                waste_count[split] += 1
        
        # Copy and remap labels
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
        
        print(f"  {split}: {waste_count[split]} images processed")
    
    # Create combined data.yaml
    data_yaml = {
        "path": str(combined_dir.absolute()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(TARGET_CLASSES),
        "names": TARGET_CLASSES
    }
    
    yaml_path = combined_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    # Calculate totals
    total_lp = sum(file_count.values())
    total_waste = sum(waste_count.values())
    
    print("\n" + "=" * 60)
    print("Dataset Merge Complete!")
    print("=" * 60)
    print(f"\nCombined data.yaml: {yaml_path}")
    print(f"\nClasses ({len(TARGET_CLASSES)}):")
    for i, cls in enumerate(TARGET_CLASSES):
        print(f"  {i}: {cls}")
    
    print(f"\nDataset Statistics:")
    print(f"  License Plate images: {total_lp}")
    print(f"  Waste Disposal images: {total_waste}")
    print(f"  --------------------------")
    print(f"  Train: {file_count['train'] + waste_count['train']}")
    print(f"  Valid: {file_count['valid'] + waste_count['valid']}")
    print(f"  Test:  {file_count['test'] + waste_count['test']}")
    print(f"  --------------------------")
    print(f"  TOTAL: {total_lp + total_waste}")
    
    return yaml_path


if __name__ == "__main__":
    create_combined_dataset()
