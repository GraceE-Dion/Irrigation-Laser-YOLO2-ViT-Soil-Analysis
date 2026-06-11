"""
Script 06: Laser crop pipeline — Phase 3
Crops laser region from each image using YOLOv5 bounding box coordinates
Phase 3 best result: 87.68% validation accuracy
Soil Moisture Classification — ViT + YOLOv8 Pipeline
"""

import os
import shutil
import yaml
import numpy as np
from PIL import Image

BASE_DIR        = '/kaggle/working/source_data'
LASER_CROP_DIR  = '/kaggle/working/Master_Laser_Crops'

if os.path.exists(LASER_CROP_DIR):
    shutil.rmtree(LASER_CROP_DIR)

mapping = {
    '0': '0',  '1': '1',  '2': '2',  '3': '3',  '4': '4',
    '5': '5',  '6': '6',  '7': '7',  '8': '8',  '9': '9', '10': '10',
    'soil-moisture-1.0': '1', 'soil-moisture-2.0': '2',
    'soil-moisture-3.0': '3', 'soil-moisture-5.0': '5',
    'soil-moisture-8.2': '8',
    'Level_0':  '0',  'Level_1':  '1',  'Level_2':  '2',
    'Level_3':  '3',  'Level_4':  '4',  'Level_5':  '5',
    'Level_6':  '6',  'Level_7':  '7',  'Level_8':  '8',
    'Level_9':  '9',  'Level_10': '10',
}

PADDING = 0.05  # 5% padding around laser region

crop_count  = 0
skip_count  = 0

for proj_folder in os.listdir(BASE_DIR):
    yaml_path = os.path.join(BASE_DIR, proj_folder, 'data.yaml')
    if not os.path.exists(yaml_path):
        continue

    with open(yaml_path, 'r') as f:
        class_names = yaml.safe_load(f)['names']

    for split in ['train', 'valid', 'test']:
        img_src = os.path.join(BASE_DIR, proj_folder, split, 'images')
        lbl_src = os.path.join(BASE_DIR, proj_folder, split, 'labels')
        target_split = 'validation' if split == 'valid' else split

        if not os.path.exists(img_src):
            continue

        for img_file in os.listdir(img_src):
            lbl_file = img_file.rsplit('.', 1)[0] + '.txt'
            lbl_p    = os.path.join(lbl_src, lbl_file)

            if not os.path.exists(lbl_p):
                skip_count += 1
                continue

            with open(lbl_p, 'r') as f:
                lines = f.readlines()
            if not lines:
                skip_count += 1
                continue

            parts      = lines[0].strip().split()
            raw_name   = str(class_names[int(parts[0])])
            clean_name = mapping.get(raw_name)
            if clean_name is None:
                skip_count += 1
                continue

            # YOLO format: class cx cy w h (normalized)
            cx, cy, w, h = float(parts[1]), float(parts[2]), \
                           float(parts[3]), float(parts[4])

            img = Image.open(os.path.join(img_src, img_file)).convert("RGB")
            W, H = img.size

            # Convert normalized coords to pixel coords with padding
            x1 = max(0, int((cx - w/2 - PADDING) * W))
            y1 = max(0, int((cy - h/2 - PADDING) * H))
            x2 = min(W, int((cx + w/2 + PADDING) * W))
            y2 = min(H, int((cy + h/2 + PADDING) * H))

            cropped = img.crop((x1, y1, x2, y2))

            dest = os.path.join(LASER_CROP_DIR, target_split, clean_name)
            os.makedirs(dest, exist_ok=True)
            unique_img = f"{proj_folder}_{img_file}"
            cropped.save(os.path.join(dest, unique_img))
            crop_count += 1

print(f"Laser crop complete: {crop_count} crops saved, {skip_count} skipped.")

# Verify counts
print("\n=== Laser Crop Dataset Counts ===")
for split in ['train', 'validation', 'test']:
    split_path = os.path.join(LASER_CROP_DIR, split)
    if os.path.exists(split_path):
        classes = os.listdir(split_path)
        total   = sum(len(os.listdir(os.path.join(split_path, c)))
                      for c in classes)
        print(f"\n{split}: {len(classes)} classes, {total} images")
        for c in sorted(classes):
            count = len(os.listdir(os.path.join(split_path, c)))
            print(f"  Class {c}: {count} images")
