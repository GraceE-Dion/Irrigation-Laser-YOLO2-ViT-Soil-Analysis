#STAGE 8: Crop Laser Regions
# Step 12: Crop Laser Regions from All Datasets
import os
import shutil
import yaml
from PIL import Image

SOURCE_DIR = '/kaggle/working/source_data'
LASER_DIR  = '/kaggle/working/Master_Laser_Crops'

# Mapping consistent with previous steps
# Mapping consistent with previous steps
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

def crop_laser(img, x_center, y_center, width, height, padding=0.05):
    """Crop laser region from image using YOLO normalized coordinates."""
    W, H = img.size
    
    # Convert normalized coords to pixel coords
    x1 = int((x_center - width/2  - padding) * W)
    y1 = int((y_center - height/2 - padding) * H)
    x2 = int((x_center + width/2  + padding) * W)
    y2 = int((y_center + height/2 + padding) * H)
    
    # Clamp to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)
    
    # If bounding box is full image, return as-is
    if width >= 0.95 and height >= 0.95:
        return img
    
    return img.crop((x1, y1, x2, y2))

if os.path.exists(LASER_DIR):
    shutil.rmtree(LASER_DIR)

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
        target_split = 'validation' if split == 'valid' else split

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

            # Parse label
            parts    = lines[0].strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width    = float(parts[3])
            height   = float(parts[4])

            raw_name   = str(class_names[class_id])
            clean_name = mapping.get(raw_name, None)
            if clean_name is None:
                skipped += 1
                continue

            # Crop and save
            img_path = os.path.join(img_dir, img_file)
            img      = Image.open(img_path).convert("RGB")
            cropped  = crop_laser(img, x_center, y_center, width, height)

            dest = os.path.join(LASER_DIR, target_split, clean_name)
            os.makedirs(dest, exist_ok=True)
            unique_name = f"{proj_folder}_{img_file}"
            cropped.save(os.path.join(dest, unique_name))
            copied += 1

print(f"Laser crops complete! {copied} saved, {skipped} skipped")
print(f"Saved to: {LASER_DIR}")

# Step 13: Verify Laser Crop Dataset
for split in ['train', 'validation', 'test']:
    split_path = os.path.join(LASER_DIR, split)
    if os.path.exists(split_path):
        classes = os.listdir(split_path)
        total   = sum(len(os.listdir(os.path.join(split_path, c))) 
                      for c in classes)
        print(f"\n{split}: {len(classes)} classes, {total} images")
        for c in sorted(classes):
            count = len(os.listdir(os.path.join(split_path, c)))
            print(f"  Class {c}: {count} images")

# Visualize sample crops
import matplotlib.pyplot as plt
import random

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

train_path = os.path.join(LASER_DIR, 'train')
all_classes = sorted(os.listdir(train_path))

for i, cls in enumerate(all_classes[:10]):
    cls_path = os.path.join(train_path, cls)
    sample   = random.choice(os.listdir(cls_path))
    img      = Image.open(os.path.join(cls_path, sample))
    axes[i].imshow(img)
    axes[i].set_title(f"Level {cls}", fontsize=12)
    axes[i].axis('off')

plt.suptitle('Sample Laser Crops by Moisture Level', fontsize=14)
plt.tight_layout()
plt.savefig('laser_crops_sample.png', dpi=150, bbox_inches='tight')
plt.show()
print("Sample crops visualized!")

# Step 14: Load Laser Crop Dataset and Train ViT

from datasets import load_dataset
from datasets import Image as HFImage

# Load cropped laser dataset
laser_ds = load_dataset(
    "imagefolder",
    data_dir=LASER_DIR,
    drop_labels=False
)
laser_ds = laser_ds.cast_column("image", HFImage(decode=True))
print(laser_ds)

# Augmentation for laser crops
from torchvision import transforms
import torch

train_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(
        brightness=0.4,   # more aggressive — laser intensity varies widely
        contrast=0.4,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomResizedCrop(
        224,
        scale=(0.8, 1.0)  # less aggressive crop — laser region is already small
    ),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
])

def transform_train(example_batch):
    augmented = [
        train_augmentation(img.convert("RGB"))
        for img in example_batch['image']
    ]
    inputs = processor(images=augmented, return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs

def transform_val(example_batch):
    inputs = processor(
        images=[img.convert("RGB") for img in example_batch['image']],
        return_tensors='pt'
    )
    inputs['labels'] = example_batch['label']
    return inputs

laser_train = laser_ds['train'].with_transform(transform_train)
laser_val   = laser_ds['validation'].with_transform(transform_val)
laser_test  = laser_ds['test'].with_transform(transform_val)

print("Laser dataset ready!")

# Step 15B — Phase 3 Extended Metrics

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             ConfusionMatrixDisplay)

# Phase 3 EXTENDED training values — 40 epochs
train_losses_p3 = [4.775843, 4.678317, 4.532147, 4.234828, 4.019403,
                   3.805770, 3.525551, 3.378486, 3.212287, 3.031305,
                   2.988930, 2.706341, 2.556252, 2.478303, 2.368372,
                   2.314680, 2.132662, 2.146253, 2.074467, 2.028231,
                   2.048648, 1.891575, 1.909996, 1.825216, 1.774547,
                   1.766110, 1.808014, 1.757672, 1.694626, 1.817509,
                   1.637901, 1.668483, 1.637347, 1.635563, 1.726993,
                   1.575352, 1.651339, 1.665032, 1.657816, 1.623149]

val_losses_p3 = [4.758066, 4.661328, 4.541890, 4.323533, 4.126867,
                 3.939586, 3.743506, 3.528446, 3.339040, 3.205756,
                 3.013563, 2.824293, 2.670386, 2.564347, 2.484676,
                 2.474155, 2.350775, 2.291309, 2.195544, 2.233965,
                 2.148098, 2.096861, 2.102739, 2.065074, 2.061624,
                 2.010844, 1.997552, 1.987275, 1.987496, 1.966841,
                 1.933921, 1.922749, 1.919895, 1.929911, 1.922580,
                 1.921657, 1.912679, 1.914333, 1.914521, 1.914572]

val_accuracies_p3 = [0.167488, 0.236453, 0.275862, 0.266010, 0.399015,
                     0.472906, 0.561576, 0.605911, 0.689655, 0.709360,
                     0.768473, 0.812808, 0.827586, 0.817734, 0.822660,
                     0.837438, 0.822660, 0.822660, 0.837438, 0.832512,
                     0.852217, 0.847291, 0.837438, 0.842365, 0.857143,
                     0.852217, 0.852217, 0.837438, 0.852217, 0.852217,
                     0.862069, 0.876847, 0.881773, 0.862069, 0.876847,
                     0.871921, 0.876847, 0.876847, 0.876847, 0.876847]

epochs_p3 = range(1, 41)  # 40 epochs
class_names = [f"Level {i}" for i in range(11)]

# Classification Report
print("\n=== PHASE 3 EXTENDED CLASSIFICATION REPORT ===")
predictions_p3 = trainer_v3.predict(laser_test)
y_pred_p3 = np.argmax(predictions_p3.predictions, axis=1)
y_true_p3 = predictions_p3.label_ids
print(classification_report(y_true_p3, y_pred_p3, target_names=class_names))

# Accuracy Graph
plt.figure(figsize=(10, 5))
plt.plot(epochs_p3, val_accuracies_p3, label='Validation Accuracy',
         marker='o', color='blue')
plt.axhline(y=0.98, color='r', linestyle='--', label='Target (98%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Phase 3 Extended — Validation Accuracy vs Target')
plt.ylim([0.1, 1.0])
plt.legend()
plt.grid(True)
plt.savefig('accuracy_graph_phase3.png', dpi=150, bbox_inches='tight')
plt.show()

# Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(epochs_p3, train_losses_p3, label='Training Loss',
         marker='o', color='blue')
plt.plot(epochs_p3, val_losses_p3, label='Validation Loss',
         marker='s', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Phase 3 Extended — Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve_phase3.png', dpi=150, bbox_inches='tight')
plt.show()

# Confusion Matrix
class_names_full = [f"Soil Moisture Level {i}" for i in range(11)]
plt.figure(figsize=(14, 12))
cm_p3 = confusion_matrix(y_true_p3, y_pred_p3)
disp  = ConfusionMatrixDisplay(confusion_matrix=cm_p3,
                                display_labels=class_names_full)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Phase 3 Extended Confusion Matrix — Laser Crop ViT Classifier')
plt.tight_layout()
plt.savefig('confusion_matrix_phase3.png', dpi=150, bbox_inches='tight')
plt.show()
print("Phase 3 extended metrics saved!")
