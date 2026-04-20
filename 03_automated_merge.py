# STAGE 3: Automated Consolidation(Check classes, consolidation, verify)
# This script scans YOLO .txt files and moves images to a Master Directory
# Step 3:Check what classes actually exist before mapping
for proj_folder in os.listdir(BASE_DIR):
    yaml_path = os.path.join(BASE_DIR, proj_folder, 'data.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        print(f"{proj_folder}: {data['names']}")

#Step 4: Consolidation and Mapping

if os.path.exists(MASTER_DIR):
    shutil.rmtree(MASTER_DIR)

# Correct mapping based on actual class names
mapping = {
    # Numeric classes - already correct
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', '10': '10',
    # Named classes from soil-moisture-5sagf and soil-moisture-ir
    'soil-moisture-1.0': '1',
    'soil-moisture-2.0': '2',
    'soil-moisture-3.0': '3',
    'soil-moisture-5.0': '5',
    'soil-moisture-8.2': '8',
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
            lbl_p = os.path.join(lbl_src, lbl_file)

            if not os.path.exists(lbl_p):
                continue

            with open(lbl_p, 'r') as f:
                lines = f.readlines()
            if not lines:
                continue

            raw_name = str(class_names[int(lines[0].split()[0])])
            clean_name = mapping.get(raw_name, None)

            if clean_name is None:
                print(f"Unmapped class: {raw_name} in {proj_folder}")
                continue

            dest = os.path.join(MASTER_DIR, target_split, clean_name)
            os.makedirs(dest, exist_ok=True)
            unique_img = f"{proj_folder}_{img_file}"
            shutil.copy(os.path.join(img_src, img_file), 
                       os.path.join(dest, unique_img))

print("Consolidation complete!")

# Step 4B: Build HuggingFace class index correction map
import os

MASTER_DIR = '/kaggle/working/Master_Soil_Moisture'

# Build correction map: HuggingFace alphabetical idx -> correct numerical idx
folders = sorted(os.listdir(os.path.join(MASTER_DIR, 'train')))
hf_to_correct = {}
for idx, folder in enumerate(folders):
    hf_to_correct[idx] = int(folder)

print("HuggingFace alphabetical index -> correct numerical class:")
for hf_idx, correct_idx in hf_to_correct.items():
    status = "✓" if hf_idx == correct_idx else "✗ FIXED"
    print(f"  hf_idx {hf_idx} -> class {correct_idx} {status}")

#Step 5: Verify Consolidation

for split in ['train', 'validation', 'test']:
    split_path = os.path.join(MASTER_DIR, split)
    if os.path.exists(split_path):
        classes = os.listdir(split_path)
        total = sum(len(os.listdir(os.path.join(split_path, c))) for c in classes)
        print(f"\n{split}: {len(classes)} classes, {total} images")
        for c in sorted(classes):
            count = len(os.listdir(os.path.join(split_path, c)))
            print(f"  Class {c}: {count} images")
