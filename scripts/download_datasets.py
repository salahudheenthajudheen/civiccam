"""
CivicCam Dataset Downloader
Downloads and merges datasets for:
1. License Plate Detection (Indian plates)
2. Waste Disposal Activity Detection
"""

import os
import shutil
import yaml
from pathlib import Path
from roboflow import Roboflow

# Configuration
API_KEY = "IhA0rRA9y7I82n2P1SZq"
BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"

# Dataset configurations
DATASETS = {
    "license_plate": {
        "workspace": "lab-gxmz0",
        "project": "alpr-indian-lwapl",
        "version": 2,
        "format": "yolov8"
    },
    "waste_disposal": {
        "workspace": "lab-gxmz0",
        "project": "waste-disposal-activity-xhtl4",
        "version": 1,
        "format": "yolov8"
    }
}


def download_dataset(rf, name, config):
    """Download a single dataset from Roboflow"""
    print(f"\n{'='*50}")
    print(f"Downloading: {name}")
    print(f"{'='*50}")
    
    try:
        project = rf.workspace(config["workspace"]).project(config["project"])
        version = project.version(config["version"])
        dataset = version.download(config["format"], location=str(DATASETS_DIR / name))
        print(f"✓ Downloaded {name} to {DATASETS_DIR / name}")
        return dataset
    except Exception as e:
        print(f"✗ Error downloading {name}: {e}")
        return None


def get_classes_from_yaml(yaml_path):
    """Extract class names from data.yaml"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', [])


def merge_datasets():
    """Merge all downloaded datasets into a combined dataset"""
    print(f"\n{'='*50}")
    print("Merging Datasets")
    print(f"{'='*50}")
    
    combined_dir = DATASETS_DIR / "combined"
    combined_train_images = combined_dir / "train" / "images"
    combined_train_labels = combined_dir / "train" / "labels"
    combined_valid_images = combined_dir / "valid" / "images"
    combined_valid_labels = combined_dir / "valid" / "labels"
    combined_test_images = combined_dir / "test" / "images"
    combined_test_labels = combined_dir / "test" / "labels"
    
    # Create directories
    for d in [combined_train_images, combined_train_labels, 
              combined_valid_images, combined_valid_labels,
              combined_test_images, combined_test_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    all_classes = []
    class_mapping = {}  # Maps old class indices to new ones per dataset
    
    # First pass: collect all unique classes
    for name in DATASETS.keys():
        dataset_dir = DATASETS_DIR / name
        yaml_path = list(dataset_dir.glob("**/data.yaml"))
        if yaml_path:
            classes = get_classes_from_yaml(yaml_path[0])
            print(f"\nClasses in {name}: {classes}")
            
            class_mapping[name] = {}
            for old_idx, class_name in enumerate(classes):
                if class_name not in all_classes:
                    all_classes.append(class_name)
                new_idx = all_classes.index(class_name)
                class_mapping[name][old_idx] = new_idx
    
    print(f"\nCombined classes: {all_classes}")
    print(f"Total classes: {len(all_classes)}")
    
    # Second pass: copy and remap files
    file_count = {"train": 0, "valid": 0, "test": 0}
    
    for name in DATASETS.keys():
        dataset_dir = DATASETS_DIR / name
        
        for split in ["train", "valid", "test"]:
            # Find images and labels directories
            images_dir = None
            labels_dir = None
            
            for subdir in dataset_dir.rglob("*"):
                if subdir.is_dir():
                    if subdir.name == "images" and split in str(subdir):
                        images_dir = subdir
                    elif subdir.name == "labels" and split in str(subdir):
                        labels_dir = subdir
            
            if not images_dir or not labels_dir:
                continue
            
            # Get destination directories
            dest_images = combined_dir / split / "images"
            dest_labels = combined_dir / split / "labels"
            
            # Copy images
            for img_file in images_dir.glob("*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    # Add prefix to avoid name collisions
                    new_name = f"{name}_{img_file.name}"
                    shutil.copy2(img_file, dest_images / new_name)
                    file_count[split] += 1
            
            # Copy and remap labels
            for label_file in labels_dir.glob("*.txt"):
                new_name = f"{name}_{label_file.name}"
                
                # Read and remap class indices
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                remapped_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_class = int(parts[0])
                        new_class = class_mapping[name].get(old_class, old_class)
                        parts[0] = str(new_class)
                        remapped_lines.append(" ".join(parts) + "\n")
                
                with open(dest_labels / new_name, 'w') as f:
                    f.writelines(remapped_lines)
    
    # Create combined data.yaml
    data_yaml = {
        "path": str(combined_dir.absolute()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(all_classes),
        "names": all_classes
    }
    
    yaml_path = combined_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n{'='*50}")
    print("Dataset Merge Complete!")
    print(f"{'='*50}")
    print(f"Combined data.yaml: {yaml_path}")
    print(f"Total classes: {len(all_classes)}")
    print(f"Classes: {all_classes}")
    print(f"Train images: {file_count['train']}")
    print(f"Valid images: {file_count['valid']}")
    print(f"Test images: {file_count['test']}")
    
    return yaml_path


def main():
    print("=" * 60)
    print("CivicCam Dataset Downloader")
    print("=" * 60)
    
    # Create datasets directory
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize Roboflow
    rf = Roboflow(api_key=API_KEY)
    
    # Download each dataset
    for name, config in DATASETS.items():
        download_dataset(rf, name, config)
    
    # Merge datasets
    yaml_path = merge_datasets()
    
    print("\n" + "=" * 60)
    print("✓ All datasets downloaded and merged successfully!")
    print(f"✓ Use this path for training: {yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
