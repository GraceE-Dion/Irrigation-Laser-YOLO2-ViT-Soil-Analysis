#STAGE 11: Phase 5: YOLOv8 Object Detection — Final Architecture
#
# Background:
# Despite five phases of ViT optimization, the best ViT result was
# 90.64% accuracy (Phase 4B). The fundamental limitation was that
# the ViT classifier — even when fed cropped laser regions — was
# treating the task as image classification rather than object
# detection. This meant the model had no explicit mechanism to
# locate and focus on the laser spot within each image.
#
# Following instructor guidance, Phase 5 reframes the entire problem
# as a single-stage object detection task using YOLOv8. Rather than
# a two-stage pipeline (detect laser, then classify), YOLOv8 detects
# the UV laser spot AND predicts the moisture level class
# simultaneously in a single forward pass. Each moisture level (0-10)
# is treated as a distinct object class, and the existing YOLOv5
# bounding box labels required zero modification for YOLOv8 training.
#
# This is the architecturally correct solution because:
# - The dataset already contains bounding box annotations
# - Moisture level is inherently tied to laser spot location
# - Object detection preserves spatial context while focusing
#   on the laser region simultaneously
# - No information is lost between detection and classification
#
# Key improvement over Phase 4B:
# - Overall mAP50 jumped from 90.64% to 95.5%
# - Level 4 mAP50 improved from 0.43 F1 to 0.935 mAP50
# - Perfect or near-perfect performance across 9 of 11 classes
# - Early stopping triggered at epoch 46, best at epoch 36
# - 5 of 7 datasets achieved perfect inference accuracy

# ──Step 22: Install YOLOv8 ────────────────────────────────────────

!pip install ultralytics -q

import ultralytics
ultralytics.checks()
print("YOLOv8 ready!")

# ──Step 23: Prepare YOLOv8 Dataset ────────────────────────────────────────

import os
import shutil
import yaml

SOURCE_DIR = '/kaggle/working/source_data'
YOLO_DIR   = '/kaggle/working/Master_YOLO'

# Class mapping — numeric only
# Class mapping — numeric only
mapping = {
    'soil-moisture-1.0': '1', 'soil-moisture-2.0': '2',
    'soil-moisture-3.0': '3', 'soil-moisture-5.0': '5',
    'soil-moisture-8.2': '8',
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', '10': '10',
    # Level_X format from september datasets
    'Level_0': '0', 'Level_1': '1', 'Level_2': '2', 'Level_3': '3',
    'Level_4': '4', 'Level_5': '5', 'Level_6': '6', 'Level_7': '7',
    'Level_8': '8', 'Level_9': '9', 'Level_10': '10',
}

if os.path.exists(YOLO_DIR):
    shutil.rmtree(YOLO_DIR)

for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(YOLO_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DIR, split, 'labels'), exist_ok=True)

skipped = 0
copied  = 0

for proj_folder in os.listdir(SOURCE_DIR):
    proj_path = os.path.join(SOURCE_DIR, proj_folder)
    yaml_path = os.path.join(proj_path, 'data.yaml')
    if not os.path.exists(yaml_path):
        continue

    with open(yaml_path, 'r') as f:
        class_names = yaml.safe_load(f)['names']

    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(proj_path, split, 'images')
        lbl_dir = os.path.join(proj_path, split, 'labels')
        target_split = 'valid' if split == 'valid' else split

        if not os.path.exists(img_dir):
            continue

        for img_file in os.listdir(img_dir):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue

            lbl_file = img_file.rsplit('.', 1)[0] + '.txt'
            lbl_path = os.path.join(lbl_dir, lbl_file)

            if not os.path.exists(lbl_path):
                skipped += 1
                continue

            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            if not lines:
                skipped += 1
                continue

            # Remap class ID
            new_lines = []
            valid = True
            for line in lines:
                parts    = line.strip().split()
                class_id = int(parts[0])
                raw_name = str(class_names[class_id])
                clean    = mapping.get(raw_name, None)
                if clean is None:
                    valid = False
                    break
                new_line = f"{clean} {' '.join(parts[1:])}\n"
                new_lines.append(new_line)

            if not valid:
                skipped += 1
                continue

            # Copy image
            unique_name = f"{proj_folder}_{img_file}"
            shutil.copy(
                os.path.join(img_dir, img_file),
                os.path.join(YOLO_DIR, target_split, 'images', unique_name)
            )

            # Write remapped label
            lbl_unique = unique_name.rsplit('.', 1)[0] + '.txt'
            with open(os.path.join(YOLO_DIR, target_split, 'labels', lbl_unique), 'w') as f:
                f.writelines(new_lines)

            copied += 1

print(f"Dataset prepared! {copied} images copied, {skipped} skipped")

# Count per split
for split in ['train', 'valid', 'test']:
    img_path = os.path.join(YOLO_DIR, split, 'images')
    if os.path.exists(img_path):
        print(f"{split}: {len(os.listdir(img_path))} images")

# ──Step 24: Create YOLOv8 data.yaml────────────────────────────────────────

import yaml

data_yaml = {
    'path': YOLO_DIR,
    'train': 'train/images',
    'val':   'valid/images',
    'test':  'test/images',
    'nc':    11,
    'names': {i: f"Level_{i}" for i in range(11)}
}

yaml_path = os.path.join(YOLO_DIR, 'data.yaml')
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print("data.yaml created!")
print(f"Location: {yaml_path}")

# Verify
with open(yaml_path, 'r') as f:
    print(yaml.safe_load(f))

# ── Step 25: Train YOLOv8 (Phase 5): Initial YOLOv8 Training ────────────────────────────────────────
# 717 training images, standard parameters
# Result: 95.5% mAP50, 81.25% inference accuracy (39/48)
# Note: september datasets had wrong class indices — results partially inflated

model_yolo = YOLO('yolov8s.pt')

results = model_yolo.train(
    data=os.path.join(YOLO_DIR, 'data.yaml'),
    epochs=50,
    imgsz=640,
    batch=16,
    name='soil_moisture_yolo',
    project='/kaggle/working/yolo_results',
    exist_ok=True,
    patience=10,
    save=True,
    plots=True,
    device=0,
    workers=2,
    lr0=0.001,
    weight_decay=0.0005,
    label_smoothing=0.1,
    val=True,
)

print("Phase 5 YOLOv8 training complete!")
print(f"Best model saved at: {results.save_dir}")


# ── Step 25: Train YOLOv8 (Phase 6)- Corrected Annotations + Class Mapping Fix ──────────────────────
# 1,026 training images (124 previously excluded september images now included)
# Fixes applied:
#   - Level_X mapping entries added to Steps 4, 12, 23
#   - HuggingFace alphabetical index correction (Step 4B)
#   - soil-moisture-8.2 ground truth display fix (Step 27)
# Result: 95.3% mAP50, 89.1% inference accuracy (41/46) — BEST MODEL

from ultralytics import YOLO

# Load pretrained YOLOv8 small model
model_yolo = YOLO('yolov8s.pt')

# Train
results = model_yolo.train(
    data=os.path.join(YOLO_DIR, 'data.yaml'),
    epochs=50,
    imgsz=640,
    batch=16,
    name='soil_moisture_yolo',
    project='/kaggle/working/yolo_results',
    exist_ok=True,
    patience=10,         # early stopping
    save=True,
    plots=True,
    device=0,            # GPU
    workers=2,
    lr0=0.001,
    weight_decay=0.0005,
    label_smoothing=0.1,
    val=True,
)

print("YOLOv8 training complete!")
print(f"Best model saved at: {results.save_dir}")

# ── Step 25: Train YOLOv8 (Phase 7)──────────────────────

from ultralytics import YOLO

# Load pretrained YOLOv8 small model
model_yolo = YOLO('yolov8s.pt')

# Train — Phase 6 parameters (best performing model)
results = model_yolo.train(
    data=os.path.join(YOLO_DIR, 'data.yaml'),
    epochs=50,
    imgsz=640,
    batch=16,
    name='soil_moisture_yolo',
    project='/kaggle/working/yolo_results',
    exist_ok=True,
    patience=10,         # early stopping
    save=True,
    plots=True,
    device=0,            # GPU
    workers=2,
    lr0=0.001,
    weight_decay=0.0005,
    label_smoothing=0.1,
    val=True,
)

# ── Step 25: Train YOLOv8 (Phase 7): Targeted Augmentation — Negative Finding ───────────────────────
# Hypothesis: aggressive hue shift (hsv_h=0.5) would simulate IR laser color
# and improve soil_moisture_stir_september classification
# Result: 93.7% mAP50, 86.9% inference accuracy (40/46) — worse than Phase 6
# Conclusion: hue-shift augmentation degraded high moisture level discrimination
# (Levels 8, 9, 10) without improving IR laser detection.
# Root cause identified through visual analysis: fundamental capture environment
# inconsistency in stir_september, not augmentation-addressable
# Phase 7 augmentation parameters (negative finding — did not improve over Phase 6)
# patience=20, hsv_h=0.5, hsv_s=0.5, hsv_v=0.4,
# fliplr=0.5, flipud=0.3, mosaic=1.0, mixup=0.2

print("YOLOv8 training complete!")
print(f"Best model saved at: {results.save_dir}")


# ── Step 26: Phase 5 — Collect YOLO auto-generated metrics ───────────────────────
import os
import shutil

yolo_results = '/kaggle/working/yolo_results/soil_moisture_yolo'

# List all auto-generated files
print("Available Phase 5 metric files:")
for root, dirs, files in os.walk(yolo_results):
    for file in files:
        filepath = os.path.join(root, file)
        print(filepath)

# Copy key metric files to working directory for easy download
key_files = [
    'confusion_matrix.png',
    'confusion_matrix_normalized.png',
    'results.png',
    'PR_curve.png',
    'F1_curve.png',
    'val_batch0_pred.jpg',
]

print("\nCopying key files to /kaggle/working/phase5_metrics/")
os.makedirs('/kaggle/working/phase5_metrics', exist_ok=True)

for file in key_files:
    src = os.path.join(yolo_results, file)
    if os.path.exists(src):
        dst = f'/kaggle/working/phase5_metrics/phase5_{file}'
        shutil.copy(src, dst)
        print(f"Copied: {file} → phase5_{file}")
    else:
        print(f"Not found: {file}")

# Zip all phase5 metrics for easy download
import zipfile
zip_path = '/kaggle/working/phase5_metrics.zip'
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for file in os.listdir('/kaggle/working/phase5_metrics'):
        zipf.write(
            os.path.join('/kaggle/working/phase5_metrics', file),
            file
        )
print(f"\nAll Phase 5 metrics zipped at: {zip_path}")

from IPython.display import FileLink
display(FileLink(zip_path))


# ── Step 26 Phase 6: Collect YOLO auto-generated metrics ───────────────────────
import os
import shutil
import zipfile
from IPython.display import FileLink, display

yolo_results = '/kaggle/working/yolo_results/soil_moisture_yolo'
output_dir   = '/kaggle/working/phase6_metrics'
zip_path     = '/kaggle/working/phase6_metrics.zip'
phase_label  = 'phase6'

# Create output directory FIRST before any file operations
os.makedirs(output_dir, exist_ok=True)

# List all auto-generated files
print(f'Available metric files:')
for root, dirs, files in os.walk(yolo_results):
    for file in files:
        print(os.path.join(root, file))

# Copy key metric files
key_files = [
    'confusion_matrix.png',
    'confusion_matrix_normalized.png',
    'results.png',
    'PR_curve.png',
    'F1_curve.png',
    'val_batch0_pred.jpg',
    'val_batch1_pred.jpg',
    'val_batch2_pred.jpg',
    'results.csv',
]

print(f"\nCopying key files to {output_dir}/")
for file in key_files:
    src = os.path.join(yolo_results, file)
    dst = os.path.join(output_dir, f'{phase_label}_{file}')
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f'Copied: {file} -> {phase_label}_{file}')
    else:
        print(f'Not found: {file}')

# Zip — directory now guaranteed to exist
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for file in os.listdir(output_dir):
        zipf.write(os.path.join(output_dir, file), arcname=file)

print(f'\nAll metrics zipped at: {zip_path}')
display(FileLink(zip_path))

# ── Step 26: Phase 7 just change the top two lines to the below ───────────────────────
# For Phase 7 change to:
output_dir  = '/kaggle/working/phase7_metrics'
zip_path    = '/kaggle/working/phase7_metrics.zip'
phase_label = 'phase7'
