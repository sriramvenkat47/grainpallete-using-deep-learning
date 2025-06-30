import os
import shutil
import random

# Set paths
SOURCE_DIR = "rice_data/Rice_Image_Dataset"
TRAIN_DIR = "data/train"
VAL_DIR = "data/validation"
SPLIT_RATIO = 0.8  # 80% train, 20% val
MAX_IMAGES_PER_CLASS = 100  # ✅ Limit for quick training

# Create output folders
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Loop through rice type folders
for cls in os.listdir(SOURCE_DIR):
    cls_path = os.path.join(SOURCE_DIR, cls)
    if not os.path.isdir(cls_path):
        continue  # Skip non-folder items

    # Make class folders in train/val
    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)

    # Get image list, shuffle, and limit to MAX_IMAGES_PER_CLASS
    files = os.listdir(cls_path)
    files = files[:MAX_IMAGES_PER_CLASS]
    random.shuffle(files)

    # Split
    split_idx = int(len(files) * SPLIT_RATIO)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    # Copy files
    for f in train_files:
        shutil.copy(os.path.join(cls_path, f), os.path.join(TRAIN_DIR, cls, f))
    for f in val_files:
        shutil.copy(os.path.join(cls_path, f), os.path.join(VAL_DIR, cls, f))

print("✅ Dataset successfully split with limited samples per class.")
