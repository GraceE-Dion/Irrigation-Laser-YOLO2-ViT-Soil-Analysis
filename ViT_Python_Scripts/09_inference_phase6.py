"""
Script 09: Phase 6 inference — annotated output images across all 7 datasets
Produces 46 annotated inference images with bounding boxes, confidence scores,
ground truth labels, and correct/incorrect indicators
Phase 6 result: 89.1% inference accuracy (41/46 correct)
Soil Moisture Classification — ViT + YOLOv8 Pipeline
"""

import os
import random
import yaml
import torch
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

SOURCE_DIR    = '/kaggle/working/source_data'
INFERENCE_DIR = '/kaggle/working/inference_phase6B'
MODEL_PATH    = '/kaggle/working/yolo_results/soil_moisture_yolo/weights/best.pt'

os.makedirs(INFERENCE_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

# Number of test images per dataset
samples_per_dataset = {
    'soil-moisture-v4':             8,
    'soil-moisture-v4-ir':          7,
    'soil-moisture-v4-uv':          7,
    'soil-moisture-ir':             7,
    'soil-moisture-5sagf':          7,
    'soil_moisture_september':      7,
    'soil_moisture_stir_september': 5,
}

# Class index mapping for ground truth labels
mapping = {
    'soil-moisture-1.0': 'Level 1', 'soil-moisture-2.0': 'Level 2',
    'soil-moisture-3.0': 'Level 3', 'soil-moisture-5.0': 'Level 5',
    'soil-moisture-8.2': 'Level 8',
    '0': 'Level 0',  '1': 'Level 1',  '2': 'Level 2',
    '3': 'Level 3',  '4': 'Level 4',  '5': 'Level 5',
    '6': 'Level 6',  '7': 'Level 7',  '8': 'Level 8',
    '9': 'Level 9',  '10': 'Level 10',
    'Level_0':  'Level 0',  'Level_1':  'Level 1',  'Level_2':  'Level 2',
    'Level_3':  'Level 3',  'Level_4':  'Level 4',  'Level_5':  'Level 5',
    'Level_6':  'Level 6',  'Level_7':  'Level 7',  'Level_8':  'Level 8',
    'Level_9':  'Level 9',  'Level_10': 'Level 10',
}

def annotate_image(img, dataset_name, img_id, pred_label, true_label, conf):
    img = img.convert("RGB").resize((640, 680))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 640, 160], fill=(0, 0, 0))
    draw.text((10,  8),  f"Dataset: {dataset_name}", fill=(255, 255, 255))
    id_display = img_id[:50] + '...' if len(img_id) > 50 else img_id
    draw.text((10, 35),  f"Image ID: {id_display}", fill=(255, 255, 255))
    pred_color = (0, 255, 0) if pred_label == true_label else (255, 0, 0)
    draw.text((10, 62),  f"Predicted:    {pred_label} ({conf:.2f})", fill=pred_color)
    draw.text((10, 89),  f"Ground Truth: {true_label}", fill=(255, 255, 0))
    result_text  = "CORRECT" if pred_label == true_label else "INCORRECT"
    result_color = (0, 255, 0) if pred_label == true_label else (255, 0, 0)
    draw.rectangle([0, 640, 640, 680], fill=(0, 0, 0))
    draw.text((10, 648), result_text, fill=result_color)
    return img

correct_total = 0
total_count   = 0

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

    ds_correct = 0
    for img_file in selected:
        img_path = os.path.join(img_dir, img_file)
        lbl_path = os.path.join(lbl_dir, img_file.rsplit('.', 1)[0] + '.txt')

        if not os.path.exists(lbl_path):
            continue

        with open(lbl_path, 'r') as f:
            lines = f.readlines()
        if not lines:
            continue

        raw_name  = str(class_names[int(lines[0].split()[0])])
        true_label = mapping.get(raw_name, raw_name)

        # Run YOLOv8 inference
        results = model(img_path, verbose=False)
        if results and results[0].boxes and len(results[0].boxes) > 0:
            best_box   = results[0].boxes[0]
            pred_cls   = int(best_box.cls.item())
            conf       = float(best_box.conf.item())
            pred_label = f"Level {pred_cls}"
        else:
            pred_label = "No detection"
            conf = 0.0

        is_correct = (pred_label == true_label)
        if is_correct:
            ds_correct   += 1
            correct_total += 1
        total_count += 1

        img = Image.open(img_path)
        annotated = annotate_image(img, dataset_name, img_file,
                                   pred_label, true_label, conf)
        out_name = f"{dataset_name}_{img_file}"
        annotated.save(os.path.join(INFERENCE_DIR, out_name))

    print(f"{dataset_name}: {ds_correct}/{len(selected)} correct "
          f"({100*ds_correct/max(len(selected),1):.1f}%)")

print(f"\nOverall: {correct_total}/{total_count} correct "
      f"({100*correct_total/max(total_count,1):.1f}%)")
