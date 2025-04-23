import glob
import os
import shutil
import numpy as np

# Define the source directory and target directories
source_dir = "data/images"
train_dir = "data/train"
val_dir = "data/val"

# Create train and validation directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get all image paths
image_paths = glob.glob(os.path.join(source_dir, "**", "*.jpg"), recursive=True)

# Split into train and validation sets (80% train, 20% val) using numpy random permutation
np.random.seed(42)
indices = np.random.permutation(len(image_paths))
split_idx = int(0.8 * len(image_paths))
train_indices = indices[:split_idx]
val_indices = indices[split_idx:]
train_paths = [image_paths[i] for i in train_indices]
val_paths = [image_paths[i] for i in val_indices]

# Copy images to train directory
for path in train_paths:
    shutil.copy(
        path, os.path.join(train_dir, path.split("/")[-2] + os.path.basename(path))
    )

# Copy images to validation directory
for path in val_paths:
    shutil.copy(
        path, os.path.join(train_dir, path.split("/")[-2] + os.path.basename(path))
    )

print(f"Training images: {len(train_paths)}, Validation images: {len(val_paths)}")
