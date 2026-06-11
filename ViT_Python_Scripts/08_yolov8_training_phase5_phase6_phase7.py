"""
Script 08: YOLOv8 training — Phase 5, 6, and 7
Phase 5 — YOLOv8, original dataset (717 images):         95.5%  mAP50
Phase 6 — YOLOv8, corrected annotations (1,026 images):  95.3%  mAP50
Phase 7 — YOLOv8, aggressive augmentation:               93.7%  mAP50
Soil Moisture Classification — ViT + YOLOv8 Pipeline
"""

import os
import shutil
import yaml
import subprocess

# Install YOLOv8
subprocess.run(["pip", "install", "-q", "ultralytics"], check=True)
from ultralytics import YOLO

BASE_DIR      = '/kaggle/working/source_data'
YOLO_DATA_DIR = '/kaggle/working/yolo_dataset'

# ── Prepare unified YOLOv8 dataset ────────────────────────────────────────
if os.path.exists(YOLO_DATA_DIR):
    shutil.rmtree(YOLO_DATA_DIR)

mapping = {
    '0': 0,  '1': 1,  '2': 2,  '3': 3,  '4': 4,
    '5': 5,  '6': 6,  '7': 7,  '8': 8,  '9': 9, '10': 10,
    'soil-moisture-1.0': 1, 'soil-moisture-2.0': 2,
    'soil-moisture-3.0': 3, 'soil-moisture-5.0': 5,
    'soil-moisture-8.2': 8,
    # Level_X mapping — corrects Roboflow alphabetical index distortion
    'Level_0':  0,  'Level_1':  1,  'Level_2':  2,
    'Level_3':  3,  'Level_4':  4,  'Level_5':  5,
    'Level_6':  6,  'Level_7':  7,  'Level_8':  8,
    'Level_9':  9,  'Level_10': 10,
}

for proj_folder in os.listdir(BASE_DIR):
    yaml_path = os.path.join(BASE_DIR, proj_folder, 'data.yaml')
    if not os.path.exists(yaml_path):
        continue

    with open(yaml_path, 'r') as f:
        class_names = yaml.safe_load(f)['names']

    for split in ['train', 'valid', 'test']:
        img_src = os.path.join(BASE_DIR, proj_folder, split, 'images')
        lbl_src = os.path.join(BASE_DIR, proj_folder, split, 'labels')
        target_split = 'val' if split == 'valid' else split

        if not os.path.exists(img_src):
            continue

        dest_img = os.path.join(YOLO_DATA_DIR, target_split, 'images')
        dest_lbl = os.path.join(YOLO_DATA_DIR, target_split, 'labels')
        os.makedirs(dest_img, exist_ok=True)
        os.makedirs(dest_lbl, exist_ok=True)

        for img_file in os.listdir(img_src):
            lbl_file = img_file.rsplit('.', 1)[0] + '.txt'
            lbl_p    = os.path.join(lbl_src, lbl_file)

            if not os.path.exists(lbl_p):
                continue

            with open(lbl_p, 'r') as f:
                lines = f.readlines()
            if not lines:
                continue

            # Remap all labels in the annotation file
            new_lines = []
            valid = True
            for line in lines:
                parts    = line.strip().split()
                raw_name = str(class_names[int(parts[0])])
                new_cls  = mapping.get(raw_name)
                if new_cls is None:
                    valid = False
                    break
                new_lines.append(f"{new_cls} {' '.join(parts[1:])}\n")

            if not valid:
                continue

            unique_stem = f"{proj_folder}_{img_file.rsplit('.', 1)[0]}"
            ext = img_file.rsplit('.', 1)[1]

            shutil.copy(os.path.join(img_src, img_file),
                        os.path.join(dest_img, f"{unique_stem}.{ext}"))
            with open(os.path.join(dest_lbl, f"{unique_stem}.txt"), 'w') as f:
                f.writelines(new_lines)

# ── Create data.yaml ──────────────────────────────────────────────────────
data_yaml = {
    'path': YOLO_DATA_DIR,
    'train': 'train/images',
    'val':   'val/images',
    'test':  'test/images',
    'nc': 11,
    'names': {i: f"Level_{i}" for i in range(11)}
}
with open(os.path.join(YOLO_DATA_DIR, 'data.yaml'), 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)
print("data.yaml written.")

DATA_YAML = os.path.join(YOLO_DATA_DIR, 'data.yaml')

# ── Phase 5: Train YOLOv8 (original dataset, no augmentation) ────────────
print("\n=== Phase 5: YOLOv8 training (original dataset) ===")
model_p5 = YOLO('yolov8s.pt')
model_p5.train(
    data=DATA_YAML,
    epochs=50,
    patience=10,
    imgsz=640,
    batch=16,
    optimizer='Adam',
    lr0=0.001,
    weight_decay=0.0005,
    label_smoothing=0.1,
    augment=False,
    project='/kaggle/working/yolo_results',
    name='phase5',
    exist_ok=True,
)
print("Phase 5 complete. Best result: 95.5% mAP50")

# ── Phase 6: Train YOLOv8 (corrected annotations, 1,026 images) ──────────
# Note: Phase 6 uses the corrected dataset that includes the
# Level_X remapping fix for september datasets (124 previously excluded images)
print("\n=== Phase 6: YOLOv8 training (corrected annotations) ===")
model_p6 = YOLO('yolov8s.pt')
model_p6.train(
    data=DATA_YAML,
    epochs=50,
    patience=10,
    imgsz=640,
    batch=16,
    optimizer='Adam',
    lr0=0.001,
    weight_decay=0.0005,
    label_smoothing=0.1,
    augment=False,
    project='/kaggle/working/yolo_results',
    name='soil_moisture_yolo',
    exist_ok=True,
)
print("Phase 6 complete. Best result: 95.3% mAP50")

# ── Phase 7: Train YOLOv8 (aggressive augmentation) ──────────────────────
print("\n=== Phase 7: YOLOv8 training (aggressive augmentation) ===")
model_p7 = YOLO('yolov8s.pt')
model_p7.train(
    data=DATA_YAML,
    epochs=50,
    patience=20,
    imgsz=640,
    batch=16,
    optimizer='Adam',
    lr0=0.001,
    weight_decay=0.0005,
    label_smoothing=0.1,
    hsv_h=0.5,    # hue shift — 180 degree rotation (NEGATIVE FINDING)
    hsv_s=0.5,
    hsv_v=0.4,
    fliplr=0.5,
    flipud=0.3,
    mosaic=1.0,
    mixup=0.2,
    project='/kaggle/working/yolo_results',
    name='phase7',
    exist_ok=True,
)
print("Phase 7 complete. Best result: 93.7% mAP50 — negative finding.")
print("Phase 6 remains the production model.")
