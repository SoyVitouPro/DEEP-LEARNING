import os
import shutil
import random

# Paths
image_dir = r"D:\Soy Vitou\GITHUB\DEEP-LEARNING\TEXT-RECOGNITION\DATASET-TESTING\dataset"
annotation_file = r"D:\Soy Vitou\GITHUB\DEEP-LEARNING\TEXT-RECOGNITION\DATASET-TESTING\annotation.txt"
output_dir = r"D:\Soy Vitou\GITHUB\DEEP-LEARNING\TEXT-RECOGNITION\DATASET-TESTING\split_dataset"

# Create output directories
train_dir = os.path.join(output_dir, "train")
valid_dir = os.path.join(output_dir, "valid")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Read and shuffle annotation data
with open(annotation_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
random.shuffle(lines)

# Split data
train_size = int(0.8 * len(lines))
valid_size = int(0.1 * len(lines))
test_size = len(lines) - train_size - valid_size

train_data = lines[:train_size]
valid_data = lines[train_size:train_size + valid_size]
test_data = lines[train_size + valid_size:]

# Function to copy images and write annotations
def process_data(data, folder, annotation_filename):
    annotation_path = os.path.join(output_dir, annotation_filename)
    with open(annotation_path, "w", encoding="utf-8") as f:
        for line in data:
            img_name, label = line.strip().split(" ", 1)
            src = os.path.join(image_dir, img_name)
            dst = os.path.join(folder, img_name)
            if os.path.exists(src):
                shutil.copy(src, dst)
                f.write(f"{img_name} {label}\n")

# Process each dataset
process_data(train_data, train_dir, "D:/Soy Vitou/GITHUB/DEEP-LEARNING/TEXT-RECOGNITION/DATASET-TESTING/dataset_split/train.txt")
process_data(valid_data, valid_dir, "D:/Soy Vitou/GITHUB/DEEP-LEARNING/TEXT-RECOGNITION/DATASET-TESTING/dataset_split/valid.txt")
process_data(test_data, test_dir, "D:/Soy Vitou/GITHUB/DEEP-LEARNING/TEXT-RECOGNITION/DATASET-TESTING/dataset_split/test.txt")

print("Dataset split complete.")