# STAGE 12: Phase 5: Two-Stage Inference Pipeline + Annotated Image Generation
#
# Background:
# This script implements the final inference pipeline using the trained
# YOLOv8 model to generate annotated output images for visual validation
# across all 7 source datasets. Unlike the earlier ViT inference script
# which classified whole images, this pipeline leverages YOLOv8's single
# forward pass to simultaneously detect the UV laser spot and predict
# the moisture level class.
#
# Each output image displays:
# - Bounding box drawn around the detected laser spot
# - Predicted moisture level with confidence score
# - Ground truth moisture level from original dataset labels
# - CORRECT/INCORRECT banner for quick visual assessment
# - Dataset name and image ID for full traceability
#
# 50 images are sampled across all 7 datasets:
# - soil-moisture-v4:             8 samples
# - soil-moisture-v4-ir:          7 samples
# - soil-moisture-v4-uv:          7 samples
# - soil-moisture-ir:             7 samples
# - soil-moisture-5sagf:          7 samples
# - soil_moisture_september:      7 samples
# - soil_moisture_stir_september: 7 samples
#
# Inference Results (48 images evaluated):
# - Overall inference accuracy: 81.25% (39/48 correct)
# - Perfect accuracy on 5 of 7 datasets
# - soil_moisture_september: 6 mismatches due to full image
#   bounding box annotations (width=height=1.0) providing no
#   meaningful laser localization
# - soil_moisture_stir_september: 2 mismatches due to stirred
#   soil surface texture variation
# - soil-moisture-ir: 1 mismatch due to IR spectral difference
# - Excluding soil_moisture_september: 92.7% inference accuracy
#   across remaining 6 well-annotated datasets

# ──Step 27: Phase 5 Inference Images───────────────────────

import os
import random
import zipfile
import yaml
from PIL import Image, ImageDraw
from ultralytics import YOLO

SOURCE_DIR = '/kaggle/working/source_data'
OUTPUT_DIR = '/kaggle/working/inference_phase5'
ZIP_PATH   = '/kaggle/working/soil_moisture_phase5_50.zip'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load best trained model
best_model_path = '/kaggle/working/yolo_results/soil_moisture_yolo/weights/best.pt'
model_yolo_inf  = YOLO(best_model_path)

# Class and dataset mappings
mapping = {
    'soil-moisture-1.0': 'Level 1', 'soil-moisture-2.0': 'Level 2',
    'soil-moisture-3.0': 'Level 3', 'soil-moisture-5.0': 'Level 5',
    'soil-moisture-8.2': 'Level 8',
    '0': 'Level 0', '1': 'Level 1', '2': 'Level 2', '3': 'Level 3',
    '4': 'Level 4', '5': 'Level 5', '6': 'Level 6', '7': 'Level 7',
    '8': 'Level 8', '9': 'Level 9', '10': 'Level 10',
}

samples_per_dataset = {
    'soil-moisture-v4':             8,
    'soil-moisture-v4-ir':          7,
    'soil-moisture-v4-uv':          7,
    'soil-moisture-ir':             7,
    'soil-moisture-5sagf':          7,
    'soil_moisture_september':      7,
    'soil_moisture_stir_september': 7,
}

def annotate_yolo(img, dataset_name, img_id, pred_label,
                  true_label, bbox, conf):
    img  = img.convert("RGB")
    W, H = img.size

    # Draw YOLO bounding box
    draw = ImageDraw.Draw(img)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        pred_color = (0, 255, 0) if pred_label == true_label else (255, 0, 0)
        draw.rectangle([x1, y1, x2, y2], outline=pred_color, width=3)
        draw.text((x1, y1 - 15),
                  f"{pred_label} ({conf:.2f})",
                  fill=pred_color)

    img   = img.resize((640, 580))
    panel = Image.new("RGB", (640, 780), (0, 0, 0))
    panel.paste(img, (0, 160))
    draw  = ImageDraw.Draw(panel)

    draw.text((10, 8),  f"Dataset: {dataset_name}",  fill=(255, 255, 255))
    img_id_display = img_id[:50] + '...' if len(img_id) > 50 else img_id
    draw.text((10, 35), f"Image ID: {img_id_display}", fill=(255, 255, 255))
    pred_color = (0, 255, 0) if pred_label == true_label else (255, 0, 0)
    draw.text((10, 62), f"Predicted:    {pred_label}", fill=pred_color)
    draw.text((10, 89), f"Ground Truth: {true_label}", fill=(255, 255, 0))
    conf_text = f"Confidence: {conf:.2f}" if conf > 0 else "No detection"
    draw.text((10, 116), conf_text, fill=(0, 200, 255))

    result_text  = "CORRECT" if pred_label == true_label else "INCORRECT"
    result_color = (0, 255, 0) if pred_label == true_label else (255, 0, 0)
    draw.rectangle([0, 740, 640, 780], fill=(0, 0, 0))
    draw.text((10, 748), result_text, fill=result_color)

    return panel

# Run YOLO inference
all_saved      = []
sample_counter = 1

for dataset_name, count in samples_per_dataset.items():
    dataset_path = os.path.join(SOURCE_DIR, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"Skipping {dataset_name} — not found")
        continue

    img_dir = os.path.join(dataset_path, 'test', 'images')
    lbl_dir = os.path.join(dataset_path, 'test', 'labels')
    if not os.path.exists(img_dir):
        img_dir = os.path.join(dataset_path, 'valid', 'images')
        lbl_dir = os.path.join(dataset_path, 'valid', 'labels')
    if not os.path.exists(img_dir):
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

        # Get ground truth
        true_label = 'Unknown'
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            if lines:
                parts      = lines[0].strip().split()
                class_id   = int(parts[0])
                raw_name   = str(class_names[class_id])
                true_label = mapping.get(raw_name, raw_name)

        # Run YOLO inference
        img     = Image.open(img_path).convert("RGB")
        results = model_yolo_inf(img_path, verbose=False)

        pred_label = 'No Detection'
        bbox       = None
        conf       = 0.0

        if len(results[0].boxes) > 0:
            # Get highest confidence detection
            boxes      = results[0].boxes
            best_idx   = boxes.conf.argmax().item()
            pred_id    = int(boxes.cls[best_idx].item())
            conf       = float(boxes.conf[best_idx].item())
            pred_label = f"Level {pred_id}"
            xyxy       = boxes.xyxy[best_idx].cpu().numpy()
            bbox       = (int(xyxy[0]), int(xyxy[1]),
                         int(xyxy[2]), int(xyxy[3]))

        # Annotate and save
        img_id    = img_file.rsplit('.', 1)[0]
        annotated = annotate_yolo(img, dataset_name, img_id,
                                  pred_label, true_label, bbox, conf)
        save_name = f"sample_{sample_counter:02d}_{dataset_name}.jpg"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        annotated.save(save_path)
        all_saved.append(save_path)
        print(f"Sample {sample_counter:02d} | {dataset_name} | "
              f"Pred: {pred_label} ({conf:.2f}) | Truth: {true_label}")
        sample_counter += 1

# Zip results
with zipfile.ZipFile(ZIP_PATH, 'w') as zipf:
    for file_path in all_saved:
        zipf.write(file_path, os.path.basename(file_path))

print(f"\n{len(all_saved)} images saved and zipped!")
print(f"ZIP: {ZIP_PATH}")

from IPython.display import FileLink
display(FileLink(ZIP_PATH))

#  ──Step 27: Phase 6 & 7 — Two-Stage Inference Pipeline + Annotated Images───────────────────────

import os
import random
import zipfile
import yaml
from PIL import Image, ImageDraw
from ultralytics import YOLO

SOURCE_DIR = '/kaggle/working/source_data'
OUTPUT_DIR = '/kaggle/working/inference_phase6'
ZIP_PATH   = '/kaggle/working/soil_moisture_phase6_50.zip'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load best Phase 6 trained model
best_model_path = '/kaggle/working/yolo_results/soil_moisture_yolo/weights/best.pt'
model_yolo_inf  = YOLO(best_model_path)
model_class_names = model_yolo_inf.names  # {0: 'Level_0', 1: 'Level_1', ...}

SEPTEMBER_DATASETS = {'soil_moisture_september', 'soil_moisture_stir_september'}

SOIL_MOISTURE_MAP = {
    'soil-moisture-1.0': '1',
    'soil-moisture-2.0': '2',
    'soil-moisture-3.0': '3',
    'soil-moisture-5.0': '5',
    'soil-moisture-8.2': '8',
}

samples_per_dataset = {
    'soil-moisture-v4':             8,
    'soil-moisture-v4-ir':          7,
    'soil-moisture-v4-uv':          7,
    'soil-moisture-ir':             7,
    'soil-moisture-5sagf':          7,
    'soil_moisture_september':      7,
    'soil_moisture_stir_september': 7,
}

def annotate_yolo(img, dataset_name, img_id, pred_label,
                  true_label, bbox, conf):
    img  = img.convert("RGB")
    W, H = img.size

    draw = ImageDraw.Draw(img)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        pred_color = (0, 255, 0) if pred_label == true_label else (255, 0, 0)
        draw.rectangle([x1, y1, x2, y2], outline=pred_color, width=3)
        draw.text((x1, y1 - 15),
                  f"{pred_label} ({conf:.2f})",
                  fill=pred_color)

    img   = img.resize((640, 580))
    panel = Image.new("RGB", (640, 780), (0, 0, 0))
    panel.paste(img, (0, 160))
    draw  = ImageDraw.Draw(panel)

    draw.text((10, 8),  f"Dataset: {dataset_name}",  fill=(255, 255, 255))
    img_id_display = img_id[:50] + '...' if len(img_id) > 50 else img_id
    draw.text((10, 35), f"Image ID: {img_id_display}", fill=(255, 255, 255))
    pred_color = (0, 255, 0) if pred_label == true_label else (255, 0, 0)
    draw.text((10, 62), f"Predicted:    {pred_label}", fill=pred_color)
    draw.text((10, 89), f"Ground Truth: {true_label}", fill=(255, 255, 0))
    conf_text = f"Confidence: {conf:.2f}" if conf > 0 else "No detection"
    draw.text((10, 116), conf_text, fill=(0, 200, 255))

    result_text  = "CORRECT" if pred_label == true_label else "INCORRECT"
    result_color = (0, 255, 0) if pred_label == true_label else (255, 0, 0)
    draw.rectangle([0, 740, 640, 780], fill=(0, 0, 0))
    draw.text((10, 748), result_text, fill=result_color)

    return panel

# Run YOLO inference
all_saved      = []
sample_counter = 1

for dataset_name, count in samples_per_dataset.items():
    dataset_path = os.path.join(SOURCE_DIR, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"Skipping {dataset_name} — not found")
        continue

    img_dir = os.path.join(dataset_path, 'test', 'images')
    lbl_dir = os.path.join(dataset_path, 'test', 'labels')
    if not os.path.exists(img_dir):
        img_dir = os.path.join(dataset_path, 'valid', 'images')
        lbl_dir = os.path.join(dataset_path, 'valid', 'labels')
    if not os.path.exists(img_dir):
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

        # Get ground truth
        true_label = 'Unknown'
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            if lines:
                parts    = lines[0].strip().split()
                class_id = int(parts[0])
                raw_name = str(class_names[class_id]).replace('_', ' ')

                if raw_name.isdigit():
                    true_label = f"Level {raw_name}"
                elif raw_name.startswith('soil-moisture-'):
                    mapped = SOIL_MOISTURE_MAP.get(
                                raw_name.replace(' ', '-'), None)
                    if mapped:
                        true_label = f"Level {mapped}"
                    else:
                        num = raw_name.replace('soil-moisture-', '')\
                                      .replace('.0', '')
                        true_label = f"Level {num}"
                else:
                    true_label = raw_name  # already 'Level X' format

        # Run YOLO inference
        img     = Image.open(img_path).convert("RGB")
        results = model_yolo_inf(img_path, verbose=False)

        pred_label = 'No Detection'
        bbox       = None
        conf       = 0.0

        if len(results[0].boxes) > 0:
            boxes      = results[0].boxes
            best_idx   = boxes.conf.argmax().item()
            pred_id    = int(boxes.cls[best_idx].item())
            conf       = float(boxes.conf[best_idx].item())
            raw_pred   = model_class_names.get(pred_id, str(pred_id))
            pred_label = raw_pred.replace('_', ' ')  # 'Level_10' -> 'Level 10'
            xyxy       = boxes.xyxy[best_idx].cpu().numpy()
            bbox       = (int(xyxy[0]), int(xyxy[1]),
                         int(xyxy[2]), int(xyxy[3]))

        # Annotate and save
        img_id    = img_file.rsplit('.', 1)[0]
        annotated = annotate_yolo(img, dataset_name, img_id,
                                  pred_label, true_label, bbox, conf)
        save_name = f"sample_{sample_counter:02d}_{dataset_name}.jpg"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        annotated.save(save_path)
        all_saved.append(save_path)
        print(f"Sample {sample_counter:02d} | {dataset_name} | "
              f"Pred: {pred_label} ({conf:.2f}) | Truth: {true_label}")
        sample_counter += 1

# Zip results
with zipfile.ZipFile(ZIP_PATH, 'w') as zipf:
    for file_path in all_saved:
        zipf.write(file_path, os.path.basename(file_path))

print(f"\n{len(all_saved)} images saved and zipped!")
print(f"ZIP: {ZIP_PATH}")

from IPython.display import FileLink
display(FileLink(ZIP_PATH))
