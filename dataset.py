import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

# Define the dataset path and target size
dataset_path = "archive/dataset-resized"
output_path = "archive/dataset-resized"
target_size = (128, 128)  # Resize images to 128x128

# Create output directories if they don't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    output_category_path = os.path.join(output_path, category)

    if not os.path.exists(output_category_path):
        os.makedirs(output_category_path)

    if os.path.isdir(category_path):
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            output_img_path = os.path.join(output_category_path, img_name)

            try:
                # Open, resize, and save the image
                with Image.open(img_path) as img:
                    img = img.resize(target_size)
                    img.save(output_img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

print("Dataset resizing complete. Resized images saved to:", output_path)

# Define paths for training and test sets
train_path = "dataset-train"
test_path = "dataset-test"
test_size = 0.2  # 20% of the data for testing

# Create directories for training and test sets
for path in [train_path, test_path]:
    if not os.path.exists(path):
        os.makedirs(path)

# Standardize and split the dataset
for category in os.listdir(output_path):
    category_path = os.path.join(output_path, category)
    if os.path.isdir(category_path):
        images = []
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                # Open and standardize the image
                with Image.open(img_path) as img:
                    img_array = (
                        np.array(img) / 255.0
                    )  # Normalize pixel values to [0, 1]
                    images.append((img_array, img_name))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        # Split into training and test sets
        train_data, test_data = train_test_split(
            images, test_size=test_size, random_state=42
        )

        # Save training data
        train_category_path = os.path.join(train_path, category)
        if not os.path.exists(train_category_path):
            os.makedirs(train_category_path)
        for img_array, img_name in train_data:
            Image.fromarray((img_array * 255).astype(np.uint8)).save(
                os.path.join(train_category_path, img_name)
            )

        # Save test data
        test_category_path = os.path.join(test_path, category)
        if not os.path.exists(test_category_path):
            os.makedirs(test_category_path)
        for img_array, img_name in test_data:
            Image.fromarray((img_array * 255).astype(np.uint8)).save(
                os.path.join(test_category_path, img_name)
            )

print("Dataset preprocessing complete. Training and test sets saved.")
