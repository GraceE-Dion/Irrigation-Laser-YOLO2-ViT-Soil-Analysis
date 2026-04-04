import os
import random
import zipfile
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from transformers import ViTForImageClassification, ViTImageProcessor

# Setup
SOURCE_DIR = '/kaggle/working/source_data'
OUTPUT_DIR = '/kaggle/working/inference_results'
ZIP_PATH = '/kaggle/working/soil_moisture_inference_50.zip'
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Class mapping
mapping = {
    'soil-moisture-1.0': 'Level 1', 'soil-moisture-2.0': 'Level 2',
    'soil-moisture-3.0': 'Level 3', 'soil-moisture-5.0': 'Level 5',
    'soil-moisture-8.2': 'Level 8',
    '0': 'Level 0', '1': 'Level 1', '2': 'Level 2', '3': 'Level 3',
    '4': 'Level 4', '5': 'Level 5', '6': 'Level 6', '7': 'Level 7',
    '8': 'Level 8', '9': 'Level 9', '10': 'Level 10',
}

# Samples per dataset
samples_per_dataset = {
    'soil-moisture-v4':             8,
    'soil-moisture-v4-ir':          7,
    'soil-moisture-v4-uv':          7,
    'soil-moisture-ir':             7,
    'soil-moisture-5sagf':          7,
    'soil_moisture_september':      7,
    'soil_moisture_stir_september': 7,
}

def annotate_image(img, dataset_name, img_id, pred_label, true_label):
    img = img.convert("RGB").resize((640, 680))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 640, 160], fill=(0, 0, 0))
    draw.text((10, 8),  f"Dataset: {dataset_name}", fill=(255, 255, 255))
    img_id_display = img_id[:50] + '...' if len(img_id) > 50 else img_id
    draw.text((10, 35), f"Image ID: {img_id_display}", fill=(255, 255, 255))
    pred_color = (0, 255, 0) if pred_label == true_label else (255, 0, 0)
    draw.text((10, 62), f"Predicted:    {pred_label}", fill=pred_color)
    draw.text((10, 89), f"Ground Truth: {true_label}", fill=(255, 255, 0))
    result_text  = "CORRECT" if pred_label == true_label else "INCORRECT"
    result_color = (0, 255, 0) if pred_label == true_label else (255, 0, 0)
    draw.rectangle([0, 640, 640, 680], fill=(0, 0, 0))
    draw.text((10, 648), result_text, fill=result_color)
    return img

# Process each dataset
all_saved = []
sample_counter = 1

for dataset_name, count in samples_per_dataset.items():
    dataset_path = os.path.join(SOURCE_DIR, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"Skipping {dataset_name} — folder not found")
        continue

    img_dir = os.path.join(dataset_path, 'test', 'images')
    lbl_dir = os.path.join(dataset_path, 'test', 'labels')
    if not os.path.exists(img_dir):
        img_dir = os.path.join(dataset_path, 'valid', 'images')
        lbl_dir = os.path.join(dataset_path, 'valid', 'labels')
    if not os.path.exists(img_dir):
        print(f"No images found for {dataset_name}")
        continue

    yaml_path = os.path.join(dataset_path, 'data.yaml')
    with open(yaml_path, 'r') as f:
        class_names = yaml.safe_load(f)['names']

    all_imgs = [f for f in os.listdir(img_dir)
                if f.endswith(('.jpg', '.jpeg', '.png'))]
    selected = random.sample(all_imgs, min(count, len(all_imgs)))

    for img_file in selected:
        img_path = os.path.join(img_dir, img_file)
        lbl_path = os.path.join(lbl_dir, img_file.rsplit('.', 1)[0] + '.txt')

        true_label = 'Unknown'
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            if lines:
                class_id  = int(lines[0].split()[0])
                raw_name  = str(class_names[class_id])
                true_label = mapping.get(raw_name, raw_name)

        # Run inference
        img    = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}  # ← FIX

        with torch.no_grad():
            outputs = model(**inputs)

        pred_id    = outputs.logits.argmax(-1).item()
        pred_label = f"Level {pred_id}"

        img_id   = img_file.rsplit('.', 1)[0]
        annotated = annotate_image(img, dataset_name, img_id, pred_label, true_label)
        save_name = f"sample_{sample_counter:02d}_{dataset_name}.jpg"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        annotated.save(save_path)
        all_saved.append(save_path)
        print(f"Sample {sample_counter:02d} | {dataset_name} | {img_id} | "
              f"Pred: {pred_label} | Truth: {true_label}")
        sample_counter += 1

# Zip all images
with zipfile.ZipFile(ZIP_PATH, 'w') as zipf:
    for file_path in all_saved:
        zipf.write(file_path, os.path.basename(file_path))

print(f"\n{len(all_saved)} images saved and zipped!")
print(f"ZIP location: {ZIP_PATH}")

from IPython.display import FileLink
display(FileLink(ZIP_PATH))

print("All files saved to /kaggle/working/ sidebar.")
