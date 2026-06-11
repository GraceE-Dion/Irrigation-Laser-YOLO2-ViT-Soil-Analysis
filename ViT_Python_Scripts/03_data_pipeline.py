"""
Script 03: Six-stage data pipeline
Stage 1 — Verify class names
Stage 2 — Consolidate and remap class labels
Stage 3 — Build HuggingFace index correction map (Stage 4B)
Stage 4 — Verify consolidation counts
Soil Moisture Classification — ViT + YOLOv8 Pipeline
"""

import os
import shutil
import yaml

BASE_DIR   = '/kaggle/working/source_data'
MASTER_DIR = '/kaggle/working/Master_Soil_Moisture'

# ── Stage 1: Check what classes exist per dataset ──────────────────────────
print("=== Stage 1: Class names per dataset ===")
for proj_folder in os.listdir(BASE_DIR):
    yaml_path = os.path.join(BASE_DIR, proj_folder, 'data.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        print(f"{proj_folder}: {data['names']}")

# ── Stage 2: Consolidation and class mapping ──────────────────────────────
print("\n=== Stage 2: Consolidation and class mapping ===")

if os.path.exists(MASTER_DIR):
    shutil.rmtree(MASTER_DIR)

# Unified mapping covering all class name formats across all seven datasets
mapping = {
    # Numeric classes — already correct
    '0': '0',  '1': '1',  '2': '2',  '3': '3',  '4': '4',
    '5': '5',  '6': '6',  '7': '7',  '8': '8',  '9': '9', '10': '10',
    # Named classes from soil-moisture-5sagf and soil-moisture-ir
    'soil-moisture-1.0': '1',
    'soil-moisture-2.0': '2',
    'soil-moisture-3.0': '3',
    'soil-moisture-5.0': '5',
    'soil-moisture-8.2': '8',
    # Level_X format from september datasets
    # (Roboflow exports alphabetically — Level_10 becomes index 2 without this fix)
    'Level_0':  '0',  'Level_1':  '1',  'Level_2':  '2',
    'Level_3':  '3',  'Level_4':  '4',  'Level_5':  '5',
    'Level_6':  '6',  'Level_7':  '7',  'Level_8':  '8',
    'Level_9':  '9',  'Level_10': '10',
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
        target_split = 'validation' if split == 'valid' else split

        if not os.path.exists(img_src):
            continue

        for img_file in os.listdir(img_src):
            lbl_file = img_file.rsplit('.', 1)[0] + '.txt'
            lbl_p    = os.path.join(lbl_src, lbl_file)

            if not os.path.exists(lbl_p):
                continue

            with open(lbl_p, 'r') as f:
                lines = f.readlines()
            if not lines:
                continue

            raw_name   = str(class_names[int(lines[0].split()[0])])
            clean_name = mapping.get(raw_name, None)

            if clean_name is None:
                print(f"Unmapped class: {raw_name} in {proj_folder}")
                continue

            dest = os.path.join(MASTER_DIR, target_split, clean_name)
            os.makedirs(dest, exist_ok=True)
            unique_img = f"{proj_folder}_{img_file}"
            shutil.copy(
                os.path.join(img_src, img_file),
                os.path.join(dest, unique_img)
            )

print("Consolidation complete.")

# ── Stage 3 (Step 4B): Build HuggingFace alphabetical index correction map ─
print("\n=== Stage 3: HuggingFace index correction map ===")
folders     = sorted(os.listdir(os.path.join(MASTER_DIR, 'train')))
hf_to_correct = {}
for idx, folder in enumerate(folders):
    hf_to_correct[idx] = int(folder)

print("HuggingFace alphabetical index -> correct numerical class:")
for hf_idx, correct_idx in hf_to_correct.items():
    status = "OK" if hf_idx == correct_idx else "FIXED"
    print(f"  hf_idx {hf_idx} -> class {correct_idx}  [{status}]")

# ── Stage 4: Verify consolidation counts ──────────────────────────────────
print("\n=== Stage 4: Consolidated dataset counts ===")
for split in ['train', 'validation', 'test']:
    split_path = os.path.join(MASTER_DIR, split)
    if os.path.exists(split_path):
        classes = os.listdir(split_path)
        total   = sum(
            len(os.listdir(os.path.join(split_path, c))) for c in classes
        )
        print(f"\n{split}: {len(classes)} classes, {total} images")
        for c in sorted(classes):
            count = len(os.listdir(os.path.join(split_path, c)))
            print(f"  Class {c}: {count} images")
