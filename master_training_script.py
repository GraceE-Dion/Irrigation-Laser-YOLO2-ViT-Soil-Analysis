# ============================================================
# FULL PIPELINE: Multi-Spectral Soil Moisture Classification
# ============================================================
# Architecture:  Vision Transformer (ViT) + YOLOv8
# Phases:        5 Development Phases
# Best Result:   95.5% mAP50 (Phase 5 — YOLOv8)
# Best ViT:      90.64% Accuracy (Phase 4B — Weighted Loss)
# Dataset:       7 Roboflow Datasets (IR, UV, Standard RGB)
# Classes:       11 Moisture Levels (0-10)
# Hardware:      Dual NVIDIA T4 GPUs
# Runtime:       8-10 hours GPU time
# ============================================================
#
# Pipeline Stages:
# Stage 1 — Data Acquisition and Consolidation
# Stage 2 — Baseline ViT Training (98.11% overfit)
# Stage 3 — Phase 1: Overfitting Fix (96.5%)
# Stage 4 — Phase 2: Data Augmentation (94.58%)
# Stage 5 — Phase 3: Laser Region Isolation (87.68%)
# Stage 6 — Phase 4A: Physical Noise Augmentation (89.66%)
# Stage 7 — Phase 4B: Class-Weighted Loss (90.64%)
# Stage 8 — Phase 5: YOLOv8 Object Detection (95.5% mAP50)
# Stage 9 — Inference and Annotated Image Generation
# ============================================================
#
# Usage:
#   Run each stage sequentially on Kaggle with GPU T4 x2 enabled
#   Required quota: minimum 10 hours GPU recommended
#   Install dependencies: pip install -r requirements.txt
#
# Results Summary:
#   Baseline:  98.11% (overfit — documented for comparison)
#   Phase 1:   96.5%  (honest baseline)
#   Phase 2:   94.58% (augmentation)
#   Phase 3:   87.68% (laser crop)
#   Phase 4A:  89.66% (noise augmentation)
#   Phase 4B:  90.64% (weighted loss — best ViT)
#   Phase 5:   95.5% mAP50 (YOLOv8 — FINAL BEST RESULT)
# ============================================================
#
# Overview:
# This script consolidates all five development phases into a single
# end-to-end executable pipeline. Running this file replicates the
# entire research project from raw data download through to final
# YOLOv8 inference and annotated image generation.
#
# Pipeline Stages:
# Stage 1 — Data Acquisition
#   Downloads all 7 Roboflow datasets (IR, UV, Standard RGB) and
#   consolidates them into a unified Master_Soil_Moisture directory
#   with consistent class mapping across all spectral modalities.
#
# Stage 2 — Baseline ViT Training
#   Trains the original whole-image Vision Transformer for 10 epochs.
#   Establishes the 98.11% baseline and documents overfitting behavior
#   for comparison against subsequent phases.
#
# Stage 3 — Phase 1: Overfitting Fix
#   Retrains ViT with dropout regularization, cosine learning rate
#   scheduling, weight decay, and early stopping. Produces an honest
#   96.5% baseline with clean loss convergence over 17 epochs.
#
# Stage 4 — Phase 2: Data Augmentation
#   Applies on-the-fly augmentation to training data only. Trains for
#   25 epochs achieving 94.58% validation accuracy with improved
#   training stability.
#
# Stage 5 — Phase 3: Laser Region Isolation
#   Extracts UV laser regions from all images using YOLOv5 bounding
#   box coordinates. Retrains ViT on cropped laser regions for 40
#   epochs achieving 87.68% accuracy with the cleanest loss curves
#   of all ViT phases.
#
# Stage 6 — Phase 4A: Physical Noise Augmentation
#   Generates Gaussian and salt-and-pepper noise copies of all
#   training images, tripling the dataset from 717 to 2,151 images.
#   Retrains for 40 epochs achieving 89.66% accuracy with Level 10
#   reaching perfect 1.00 F1 score.
#
# Stage 7 — Phase 4B: Class-Weighted Loss
#   Implements inverse frequency class weighting through a custom
#   WeightedTrainer targeting weak classes (Levels 2, 4, 6).
#   Achieves 90.64% accuracy — best ViT result across all phases.
#
# Stage 8 — Phase 5: YOLOv8 Object Detection
#   Trains YOLOv8s on the original dataset labels treating each
#   moisture level as a distinct object class. Achieves 95.5% mAP50
#   — the best result across all phases — by detecting and classifying
#   the UV laser spot simultaneously in a single forward pass.
#
# Stage 9 — Inference and Visualization
#   Runs the trained YOLOv8 model on 50 unseen images across all 7
#   datasets, generating annotated output images showing bounding
#   boxes, predicted moisture levels, confidence scores, and ground
#   truth labels for direct visual comparison.
#
# Usage:
#   Run each stage sequentially on Kaggle with GPU T4 x2 enabled.
#   Estimated total runtime: 8-10 hours GPU time.
#   Required GPU quota: minimum 10 hours recommended.
#
# Requirements:
#   pip install -r requirements.txt
#
# Results Summary:
#   Baseline:  98.11% (overfit)
#   Phase 1:   96.5%
#   Phase 2:   94.58%
#   Phase 3:   87.68%
#   Phase 4A:  89.66%
#   Phase 4B:  90.64%
#   Phase 5:   95.5% mAP50 — FINAL BEST RESULT

# STAGE 1: Environment Setup
# Installs necessary AI libraries for Vision Transformers and Data Handling
!pip install -q evaluate roboflow datasets transformers[torch]

import os, shutil, yaml, torch, numpy as np, evaluate
from roboflow import Roboflow
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    DefaultDataCollator,
    Trainer
)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# STAGE 2: Multi-Source Data Acquisition
rf = Roboflow(api_key="yRqyBbimhh1vgoeZs2Gx")

projects = [
    ("robotics-lab-1", "soil-moisture-v4", 3),
    ("robotics-lab-1", "soil-moisture-v4-ir", 1),
    ("robotics-lab-1", "soil-moisture-v4-uv", 1),
    ("robotics-lab-1", "soil-moisture-ir", 1),
    ("robotics-lab-1", "soil-moisture-5sagf", 1),
    ("robotics-lab-1", "soil_moisture_september", 4),
    ("robotics-lab-1", "soil_moisture_stir_september", 1)
]

BASE_DIR = '/kaggle/working/source_data'
MASTER_DIR = '/kaggle/working/Master_Soil_Moisture'
os.makedirs(BASE_DIR, exist_ok=True)

for workspace, proj_name, ver in projects:
    try:
        project = rf.workspace(workspace).project(proj_name)
        dataset = project.version(ver).download("yolov5", 
                  location=os.path.join(BASE_DIR, proj_name))
    except Exception as e:
        print(f"Skipping {proj_name}: {e}")

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

# STAGE 4: Feature Extraction(Load raw-ds, processor, transforms)
#Step 6: Load Datasets and Check Column Names

from datasets import load_dataset, Image as HFImage

raw_ds = load_dataset(
    "imagefolder",
    data_dir=MASTER_DIR,
    drop_labels=False
)

raw_ds = raw_ds.cast_column("image", HFImage(decode=True))
print(raw_ds)

#Step 7: Defining Processor

from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
print("Processor loaded!")

# Step 8: Transform

# Original whole-image transform for baseline and Phase 1

from PIL import Image as PILImage
from transformers import ViTImageProcessor

# Load processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
print("Processor loaded!")

def transform(example_batch):
    inputs = processor(
        [x.convert("RGB") for x in example_batch['image']],
        return_tensors='pt'
    )
    inputs['labels'] = example_batch['label']
    return inputs

prepared_ds = raw_ds.with_transform(transform)
print("Dataset transformed and ready!")

# STAGE 5: Vision Transformer Fine-Tuning
# Original ViT Baseline Training — 10 epochs (overfit)

import evaluate
import numpy as np
from transformers import (
    ViTForImageClassification,
    TrainingArguments,
    DefaultDataCollator,
    Trainer
)

# Model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=11,
    id2label={i: f"Level {i}" for i in range(11)},
    label2id={f"Level {i}": i for i in range(11)},
    ignore_mismatched_sizes=True
)

# Metric
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# Training Arguments — Original Baseline
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=5,
    num_train_epochs=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    remove_unused_columns=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_ds['train'],
    eval_dataset=prepared_ds['validation'],
    processing_class=processor,
    compute_metrics=compute_metrics
)

trainer.train()

# Save model
trainer.save_model('./results/final_model')
processor.save_pretrained('./results/final_model')
print("Baseline model saved!")

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Exact values from baseline training output
train_losses = [1.756171, 1.077581, 0.757115, 0.496666,
                0.378825, 0.330151, 0.301320, 0.278144,
                0.261974, 0.273489]

val_losses = [1.570183, 1.042456, 0.747440, 0.591142,
              0.498698, 0.443029, 0.411059, 0.391569,
              0.379718, 0.376136]

val_accuracies = [0.871921, 0.931034, 0.940887, 0.955665,
                  0.965517, 0.965517, 0.970443, 0.970443,
                  0.970443, 0.970443]

epochs = range(1, 11)
class_names = [f"Level {i}" for i in range(11)]

# Classification Report
print("\n=== BASELINE CLASSIFICATION REPORT ===")
predictions = trainer.predict(prepared_ds['test'])
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids
print(classification_report(y_true, y_pred, target_names=class_names))

# Accuracy Graph
plt.figure(figsize=(10, 5))
plt.plot(epochs, val_accuracies, label='Validation Accuracy',
         marker='o', color='blue')
plt.axhline(y=0.98, color='r', linestyle='--', label='Target (98%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Baseline — Validation Accuracy vs Target')
plt.ylim([0.8, 1.0])
plt.legend()
plt.grid(True)
plt.savefig('accuracy_graph_baseline.png', dpi=150, bbox_inches='tight')
plt.show()
print("Accuracy graph saved!")

# Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label='Training Loss',
         marker='o', color='blue')
plt.plot(epochs, val_losses, label='Validation Loss',
         marker='s', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Baseline — Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve_baseline.png', dpi=150, bbox_inches='tight')
plt.show()
print("Loss curve saved!")

# Confusion Matrix
class_names_full = [f"Soil Moisture Level {i}" for i in range(11)]
plt.figure(figsize=(14, 12))
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=class_names_full)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Baseline Confusion Matrix — Soil Moisture ViT Classifier')
plt.tight_layout()
plt.savefig('confusion_matrix_baseline.png', dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix saved!")

# STAGE 6: Validation & Results
# Generates the Confusion Matrix, Accuracy graph, loss curve and Classification Report
# Phase 1: Overfitting Fix — 17 epochs, dropout, early stopping

import evaluate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import (
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# Model with dropout regularization
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=11,
    id2label={i: f"Level {i}" for i in range(11)},
    label2id={f"Level {i}": i for i in range(11)},
    ignore_mismatched_sizes=True,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)

# Metric
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# Training Arguments — Phase 1
training_args = TrainingArguments(
    output_dir="./results_v2",
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=5,
    num_train_epochs=20,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    remove_unused_columns=False,
    save_total_limit=1,
)

# Trainer with Early Stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_ds['train'],
    eval_dataset=prepared_ds['validation'],
    processing_class=processor,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )]
)

trainer.train()

# Save model
trainer.save_model('./results_v2/final_model')
processor.save_pretrained('./results_v2/final_model')
print("Phase 1 model saved!")

# Phase 1 actual training values — 17 epochs
train_losses_p1 = [2.344809, 2.118603, 1.936597, 1.603722, 1.359488,
                   1.207903, 1.005909, 0.903221, 0.784347, 0.738066,
                   0.660259, 0.600291, 0.564556, 0.526923, 0.508441,
                   0.523409, 0.499586]

val_losses_p1 = [2.318297, 2.081824, 1.797832, 1.574416, 1.370550,
                 1.195082, 1.046510, 0.922475, 0.825795, 0.748198,
                 0.666923, 0.640123, 0.605296, 0.580547, 0.567832,
                 0.554361, 0.547682]

val_accuracies_p1 = [0.270936, 0.448276, 0.674877, 0.862069, 0.871921,
                     0.886700, 0.940887, 0.950739, 0.950739, 0.955665,
                     0.960591, 0.960591, 0.960591, 0.965517, 0.965517,
                     0.965517, 0.965517]

epochs_p1 = range(1, 18)
class_names = [f"Level {i}" for i in range(11)]

# Classification Report
print("\n=== PHASE 1 CLASSIFICATION REPORT ===")
predictions = trainer.predict(prepared_ds['test'])
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids
print(classification_report(y_true, y_pred, target_names=class_names))

# Accuracy Graph
plt.figure(figsize=(10, 5))
plt.plot(epochs_p1, val_accuracies_p1, label='Validation Accuracy',
         marker='o', color='blue')
plt.axhline(y=0.98, color='r', linestyle='--', label='Target (98%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Phase 1 — Validation Accuracy vs Target')
plt.ylim([0.2, 1.0])
plt.legend()
plt.grid(True)
plt.savefig('accuracy_graph_phase1.png', dpi=150, bbox_inches='tight')
plt.show()
print("Phase 1 accuracy graph saved!")

# Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(epochs_p1, train_losses_p1, label='Training Loss',
         marker='o', color='blue')
plt.plot(epochs_p1, val_losses_p1, label='Validation Loss',
         marker='s', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Phase 1 — Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve_phase1.png', dpi=150, bbox_inches='tight')
plt.show()
print("Phase 1 loss curve saved!")

# Confusion Matrix
class_names_full = [f"Soil Moisture Level {i}" for i in range(11)]
plt.figure(figsize=(14, 12))
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=class_names_full)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Phase 1 Confusion Matrix — Soil Moisture ViT Classifier')
plt.tight_layout()
plt.savefig('confusion_matrix_phase1.png', dpi=150, bbox_inches='tight')
plt.show()
print("Phase 1 confusion matrix saved!")

#STAGE 7
#the training arguments was changed to address the overfitting of the model by 
#retraining the existing whole-image ViT with the regularization fixes, 
#so you have a clean baseline accuracy improvement to compare against

# Step 8 REVISED — Fixed Augmentation

from PIL import Image as PILImage
from torchvision import transforms
import torch

# Augmentation ONLY — no ToTensor or Normalize here
train_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomResizedCrop(
        224,
        scale=(0.7, 1.0),
        ratio=(0.8, 1.2)
    ),
    transforms.GaussianBlur(
        kernel_size=3,
        sigma=(0.1, 2.0)
    ),
    transforms.RandomAdjustSharpness(
        sharpness_factor=2,
        p=0.3
    ),
    # NO ToTensor or Normalize — processor handles this
])

def transform_train(example_batch):
    augmented_images = [
        train_augmentation(img.convert("RGB")) 
        for img in example_batch['image']
    ]
    inputs = processor(
        images=augmented_images,
        return_tensors='pt'
    )
    inputs['labels'] = example_batch['label']
    return inputs

def transform_val(example_batch):
    inputs = processor(
        images=[img.convert("RGB") for img in example_batch['image']],
        return_tensors='pt'
    )
    inputs['labels'] = example_batch['label']
    return inputs

# Apply transforms
prepared_ds_train = raw_ds['train'].with_transform(transform_train)
prepared_ds_val   = raw_ds['validation'].with_transform(transform_val)
prepared_ds_test  = raw_ds['test'].with_transform(transform_val)

print("Augmentation pipeline ready!")

# Step 9 REVISED — Phase 2: Training with Augmentation
import evaluate
import numpy as np
from transformers import (
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# Model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=11,
    id2label={i: f"Level {i}" for i in range(11)},
    label2id={f"Level {i}": i for i in range(11)},
    ignore_mismatched_sizes=True,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)

# Metric
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# Training Arguments — Phase 2 UPDATED
training_args = TrainingArguments(
    output_dir="./results_v3",
    save_total_limit=1,
    save_strategy="no",              # no checkpoints during training
    load_best_model_at_end=False,    # must be False when save_strategy="no"
    eval_strategy="epoch",
    logging_steps=5,
    num_train_epochs=25,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    metric_for_best_model="accuracy",
    greater_is_better=True,
    remove_unused_columns=False,
    label_smoothing_factor=0.1,
)

# Trainer — no EarlyStoppingCallback since save_strategy="no"
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_ds_train,
    eval_dataset=prepared_ds_val,
    processing_class=processor,
    compute_metrics=compute_metrics,
)
trainer.train()
# Save once after training completes
trainer.save_model('./results_v3/final_model')
processor.save_pretrained('./results_v3/final_model')
print("Model saved!")

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

#STAGE 9: Physical Noise Augmentation on Laser Crops
#
# Background:
# Phase 3 achieved 87.68% accuracy using cropped laser regions but
# mid-range moisture levels (3, 4, 6) showed persistent confusion.
# Following instructor guidance, Phase 4A implements physical noise
# augmentation — generating and saving Gaussian noise and salt-and-pepper
# noise copies of each training image to disk, effectively tripling the
# training set from 717 to 2,151 images.
#
# Unlike on-the-fly augmentation used in Phase 2, this approach physically
# creates new image files that are saved alongside the originals before
# reloading the dataset. This ensures the model trains on the combined
# original and noisy data together, exposing it to a wider variety of
# laser diffusion patterns across different noise conditions.
#
# Key improvement over Phase 3:
# - Training set tripled from 717 to 2,151 images
# - Level 10 F1 score improved from 0.80 to 1.00
# - Level 2 F1 score improved from 0.64 to 0.80
# - Overall accuracy improved from 87.68% to 89.66%

# Step 16 — Phase 4A: Generate Noisy Augmented Images
import os
import numpy as np
from PIL import Image
import random

LASER_DIR = '/kaggle/working/Master_Laser_Crops'
train_path = os.path.join(LASER_DIR, 'train')

def add_gaussian_noise(img, mean=0, std=25):
    """Add Gaussian noise to image."""
    img_array = np.array(img).astype(np.float32)
    noise = np.random.normal(mean, std, img_array.shape)
    noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def add_salt_pepper_noise(img, amount=0.05):
    """Add salt and pepper noise to image."""
    img_array = np.array(img).astype(np.uint8)
    noisy = img_array.copy()
    
    # Salt — white pixels
    num_salt = int(amount * img_array.size * 0.5)
    salt_coords = [np.random.randint(0, i, num_salt) 
                   for i in img_array.shape[:2]]
    noisy[salt_coords[0], salt_coords[1]] = 255
    
    # Pepper — black pixels
    num_pepper = int(amount * img_array.size * 0.5)
    pepper_coords = [np.random.randint(0, i, num_pepper) 
                     for i in img_array.shape[:2]]
    noisy[pepper_coords[0], pepper_coords[1]] = 0
    
    return Image.fromarray(noisy)

def flip_image(img):
    """Apply random horizontal or vertical flip."""
    choice = random.randint(0, 2)
    if choice == 0:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif choice == 1:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        # Both flips
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img.transpose(Image.FLIP_TOP_BOTTOM)

# Generate augmented copies
augmented_count = 0

for class_folder in os.listdir(train_path):
    class_path = os.path.join(train_path, class_folder)
    if not os.path.isdir(class_path):
        continue

    original_files = [f for f in os.listdir(class_path)
                      if f.endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in original_files:
        img_path = os.path.join(class_path, img_file)
        img = Image.open(img_path).convert("RGB")
        base_name = img_file.rsplit('.', 1)[0]

        # Copy 1 — flip + Gaussian noise
        aug1 = flip_image(img)
        aug1 = add_gaussian_noise(aug1, mean=0, std=25)
        aug1_path = os.path.join(class_path, f"{base_name}_aug_gaussian.jpg")
        aug1.save(aug1_path)

        # Copy 2 — flip + Salt & Pepper noise
        aug2 = flip_image(img)
        aug2 = add_salt_pepper_noise(aug2, amount=0.05)
        aug2_path = os.path.join(class_path, f"{base_name}_aug_saltpepper.jpg")
        aug2.save(aug2_path)

        augmented_count += 2

print(f"Augmentation complete!")
print(f"Original images: 717")
print(f"New augmented images added: {augmented_count}")
print(f"Total training images: {717 + augmented_count}")

# Verify counts per class
print("\nPer class breakdown:")
for class_folder in sorted(os.listdir(train_path)):
    class_path = os.path.join(train_path, class_folder)
    if os.path.isdir(class_path):
        count = len(os.listdir(class_path))
        print(f"  Class {class_folder}: {count} images")

Step 17: Reload Expanded Dataset

from datasets import load_dataset
from datasets import Image as HFImage

# Reload laser dataset — now includes augmented images
laser_ds_aug = load_dataset(
    "imagefolder",
    data_dir=LASER_DIR,
    drop_labels=False
)
laser_ds_aug = laser_ds_aug.cast_column("image", HFImage(decode=True))
print(laser_ds_aug)
print(f"Training set expanded to: {laser_ds_aug['train'].num_rows} images")

Step 18: Apply Transforms to Expanded Dataset

from datasets import load_dataset
from datasets import Image as HFImage

# Reload laser dataset — now includes augmented images
laser_ds_aug = load_dataset(
    "imagefolder",
    data_dir=LASER_DIR,
    drop_labels=False
)
laser_ds_aug = laser_ds_aug.cast_column("image", HFImage(decode=True))
print(laser_ds_aug)
print(f"Training set expanded to: {laser_ds_aug['train'].num_rows} images")

Step 19: Train ViT on Augmented Laser Crops

import evaluate
import numpy as np
from transformers import (
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)

# Fresh model
model_v4 = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=11,
    id2label={i: f"Level {i}" for i in range(11)},
    label2id={f"Level {i}": i for i in range(11)},
    ignore_mismatched_sizes=True,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

training_args_v4 = TrainingArguments(
    output_dir="./results_v5",
    save_strategy="no",
    eval_strategy="epoch",
    logging_steps=5,
    num_train_epochs=40,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    load_best_model_at_end=False,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    remove_unused_columns=False,
    label_smoothing_factor=0.1,
    save_total_limit=1,
)

trainer_v4 = Trainer(
    model=model_v4,
    args=training_args_v4,
    train_dataset=laser_train_aug,
    eval_dataset=laser_val_aug,
    processing_class=processor,
    compute_metrics=compute_metrics,
)

trainer_v4.train()
# Save final model
trainer_v4.save_model('./results_v5/final_model')
processor.save_pretrained('./results_v5/final_model')
print("Phase 4A model saved!")

# Step 20A: Phase 4A Metrics
# Fix — import missing and replot confusion matrix only
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class_names_full = [f"Soil Moisture Level {i}" for i in range(11)]
plt.figure(figsize=(14, 12))
cm_p4a = confusion_matrix(y_true_p4a, y_pred_p4a)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_p4a,
                               display_labels=class_names_full)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Phase 4A Confusion Matrix — Augmented Laser Crop ViT Classifier')
plt.tight_layout()
plt.savefig('confusion_matrix_phase4a.png', dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix saved!")
train_losses_p4a = [4.598878, 4.018516, 3.466031, 3.105857, 2.701540,
                    2.416365, 2.183443, 1.930872, 1.865558, 1.660936,
                    1.601121, 1.554153, 1.487123, 1.435534, 1.452652,
                    1.371130, 1.396241, 1.277487, 1.367294, 1.145833,
                    1.172555, 1.357800, 1.288642, 1.175231, 1.273762,
                    1.203880, 1.224792, 1.190668, 1.156583, 1.123374,
                    1.139964, 1.203056, 1.173595, 1.167897, 1.171117,
                    1.172913, 1.196974, 1.106842, 1.134756, 1.133870]

val_losses_p4a = [4.553881, 3.976180, 3.464017, 2.997657, 2.627634,
                  2.384494, 2.262010, 2.055072, 1.935329, 1.903518,
                  1.802516, 1.753705, 1.762875, 1.825698, 1.669055,
                  1.766719, 1.687311, 1.614212, 1.633559, 1.572980,
                  1.654490, 1.615139, 1.625992, 1.633548, 1.589483,
                  1.614638, 1.604377, 1.591931, 1.569810, 1.570749,
                  1.570167, 1.573008, 1.596105, 1.591208, 1.593277,
                  1.589233, 1.593882, 1.597016, 1.597456, 1.597225]

val_accuracies_p4a = [0.216749, 0.344828, 0.566502, 0.778325, 0.812808,
                      0.812808, 0.778325, 0.842365, 0.842365, 0.822660,
                      0.842365, 0.852217, 0.857143, 0.837438, 0.876847,
                      0.847291, 0.871921, 0.886700, 0.876847, 0.896552,
                      0.871921, 0.876847, 0.891626, 0.876847, 0.886700,
                      0.876847, 0.881773, 0.891626, 0.896552, 0.891626,
                      0.881773, 0.891626, 0.886700, 0.881773, 0.881773,
                      0.886700, 0.886700, 0.881773, 0.886700, 0.886700]

epochs_p4a = range(1, 41)
class_names = [f"Level {i}" for i in range(11)]

# Classification Report
print("\n=== PHASE 4A CLASSIFICATION REPORT ===")
predictions_p4a = trainer_v4.predict(laser_test_aug)
y_pred_p4a = np.argmax(predictions_p4a.predictions, axis=1)
y_true_p4a = predictions_p4a.label_ids
print(classification_report(y_true_p4a, y_pred_p4a, target_names=class_names))

# Accuracy Graph
plt.figure(figsize=(10, 5))
plt.plot(epochs_p4a, val_accuracies_p4a, label='Validation Accuracy',
         marker='o', color='blue')
plt.axhline(y=0.98, color='r', linestyle='--', label='Target (98%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Phase 4A — Validation Accuracy vs Target')
plt.ylim([0.1, 1.0])
plt.legend()
plt.grid(True)
plt.savefig('accuracy_graph_phase4a.png', dpi=150, bbox_inches='tight')
plt.show()

# Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(epochs_p4a, train_losses_p4a, label='Training Loss',
         marker='o', color='blue')
plt.plot(epochs_p4a, val_losses_p4a, label='Validation Loss',
         marker='s', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Phase 4A — Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve_phase4a.png', dpi=150, bbox_inches='tight')
plt.show()

# Confusion Matrix
class_names_full = [f"Soil Moisture Level {i}" for i in range(11)]
plt.figure(figsize=(14, 12))
cm_p4a = confusion_matrix(y_true_p4a, y_pred_p4a)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_p4a,
                               display_labels=class_names_full)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Phase 4A Confusion Matrix — Augmented Laser Crop ViT Classifier')
plt.tight_layout()
plt.savefig('confusion_matrix_phase4a.png', dpi=150, bbox_inches='tight')
plt.show()
print("Phase 4A metrics saved!")

# STAGE 10:Phase 4B: Class-Weighted Loss Function
#
# Background:
# Phase 4A improved overall accuracy to 89.66% through physical noise
# augmentation, and Level 10 reached a perfect 1.00 F1 score. However
# Level 4 unexpectedly dropped to 0.25 F1 and Level 6 remained weak
# at 0.52 F1. Analysis of the confusion matrix revealed that Level 4
# was being heavily confused with its neighbors Levels 3 and 5, with
# 6 out of 9 test samples misclassified.
#
# Phase 4B addresses this class imbalance by implementing inverse
# frequency class weighting through a custom WeightedTrainer. Class
# weights are computed from training sample counts — underrepresented
# classes receive higher weights, forcing the model to pay more
# attention to difficult classes during training.
#
# Targeted weak classes:
# - Level 2 (78 training samples) — weight 2.0
# - Level 4 (109 training samples) — weight 3.0
# - Level 6 (42 training samples) — weight 2.0
#
# Key improvement over Phase 4A:
# - Overall accuracy improved from 89.66% to 90.64%
# - Level 4 F1 recovered from 0.25 to 0.43
# - Level 6 F1 recovered from 0.52 to 0.64
# - Level 10 remained perfect at 1.00
# - Best ViT result across all phases

#Step 19B: Weighted Trainer
import torch
import numpy as np
import evaluate
from transformers import ViTForImageClassification, TrainingArguments, Trainer

# Compute class weights from training data
class_counts = [34, 59, 78, 65, 109, 105, 42, 53, 65, 36, 71]  # from Step 13 output
total = sum(class_counts)
num_classes = len(class_counts)

# Inverse frequency weighting — rare classes get higher weight
class_weights = torch.tensor(
    [total / (num_classes * count) for count in class_counts],
    dtype=torch.float32
).to('cuda')

print("Class weights:")
for i, w in enumerate(class_weights):
    print(f"  Level {i}: {w:.4f}")

# Custom Trainer with weighted loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Fresh model
model_v4b = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=11,
    id2label={i: f"Level {i}" for i in range(11)},
    label2id={f"Level {i}": i for i in range(11)},
    ignore_mismatched_sizes=True,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

training_args_v4b = TrainingArguments(
    output_dir="./results_v6",
    save_strategy="no",
    eval_strategy="epoch",
    logging_steps=5,
    num_train_epochs=40,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    load_best_model_at_end=False,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    remove_unused_columns=False,
    label_smoothing_factor=0.1,
    save_total_limit=1,
)

trainer_v4b = WeightedTrainer(
    model=model_v4b,
    args=training_args_v4b,
    train_dataset=laser_train_aug,
    eval_dataset=laser_val_aug,
    processing_class=processor,
    compute_metrics=compute_metrics,
)

trainer_v4b.train()

# Save
trainer_v4b.save_model('./results_v6/final_model')
processor.save_pretrained('./results_v6/final_model')
print("Phase 4B weighted loss model saved!")

# Step 20B: Phase 4B Metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             ConfusionMatrixDisplay)

train_losses_p4b = [2.293143, 1.928957, 1.661045, 1.403453, 1.121677,
                    1.010281, 0.819675, 0.739883, 0.671993, 0.521686,
                    0.500440, 0.470915, 0.441232, 0.418277, 0.377515,
                    0.331545, 0.337027, 0.261187, 0.324425, 0.231847,
                    0.224760, 0.282517, 0.225149, 0.176197, 0.212028,
                    0.188740, 0.206023, 0.179401, 0.159163, 0.140444,
                    0.139795, 0.232164, 0.192412, 0.164299, 0.171826,
                    0.165090, 0.184149, 0.133453, 0.153742, 0.133313]

val_losses_p4b = [2.288795, 2.021616, 1.686626, 1.453947, 1.222767,
                  1.093630, 0.958269, 0.834218, 0.782016, 0.725853,
                  0.660369, 0.623775, 0.587952, 0.634187, 0.525539,
                  0.503068, 0.516293, 0.485166, 0.499607, 0.514354,
                  0.516399, 0.491443, 0.452487, 0.433715, 0.431894,
                  0.433852, 0.426308, 0.426420, 0.435224, 0.420108,
                  0.414563, 0.408872, 0.418126, 0.415748, 0.416119,
                  0.408963, 0.416059, 0.415985, 0.415391, 0.415321]

val_accuracies_p4b = [0.221675, 0.379310, 0.566502, 0.640394, 0.640394,
                      0.699507, 0.773399, 0.827586, 0.822660, 0.822660,
                      0.847291, 0.852217, 0.857143, 0.822660, 0.871921,
                      0.876847, 0.871921, 0.881773, 0.871921, 0.862069,
                      0.866995, 0.862069, 0.876847, 0.901478, 0.891626,
                      0.896552, 0.891626, 0.896552, 0.896552, 0.896552,
                      0.901478, 0.906404, 0.896552, 0.896552, 0.901478,
                      0.906404, 0.901478, 0.901478, 0.901478, 0.901478]

epochs_p4b = range(1, 41)
class_names = [f"Level {i}" for i in range(11)]

# Classification Report
print("\n=== PHASE 4B CLASSIFICATION REPORT ===")
predictions_p4b = trainer_v4b.predict(laser_test_aug)
y_pred_p4b = np.argmax(predictions_p4b.predictions, axis=1)
y_true_p4b = predictions_p4b.label_ids
print(classification_report(y_true_p4b, y_pred_p4b, target_names=class_names))

# Accuracy Graph
plt.figure(figsize=(10, 5))
plt.plot(epochs_p4b, val_accuracies_p4b, label='Validation Accuracy',
         marker='o', color='blue')
plt.axhline(y=0.98, color='r', linestyle='--', label='Target (98%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Phase 4B — Validation Accuracy vs Target')
plt.ylim([0.1, 1.0])
plt.legend()
plt.grid(True)
plt.savefig('accuracy_graph_phase4b.png', dpi=150, bbox_inches='tight')
plt.show()

# Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(epochs_p4b, train_losses_p4b, label='Training Loss',
         marker='o', color='blue')
plt.plot(epochs_p4b, val_losses_p4b, label='Validation Loss',
         marker='s', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Phase 4B — Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve_phase4b.png', dpi=150, bbox_inches='tight')
plt.show()

# Confusion Matrix
class_names_full = [f"Soil Moisture Level {i}" for i in range(11)]
plt.figure(figsize=(14, 12))
cm_p4b = confusion_matrix(y_true_p4b, y_pred_p4b)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_p4b,
                               display_labels=class_names_full)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Phase 4B Confusion Matrix — Weighted Loss ViT Classifier')
plt.tight_layout()
plt.savefig('confusion_matrix_phase4b.png', dpi=150, bbox_inches='tight')
plt.show()
print("Phase 4B metrics saved!")

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

# Step 22: Install YOLOv8

!pip install ultralytics -q

import ultralytics
ultralytics.checks()
print("YOLOv8 ready!")

# Step 23: Prepare YOLOv8 Dataset

import os
import shutil
import yaml

SOURCE_DIR = '/kaggle/working/source_data'
YOLO_DIR   = '/kaggle/working/Master_YOLO'

# Class mapping — numeric only
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

# Step 24: Create YOLOv8 data.yaml

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

# Step 25: Train YOLOv8(Phase 6)

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

# Step 25: Train YOLOv8 (Phase 7)
# Step 25: Train YOLOv8
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

# Phase 7 augmentation parameters (negative finding — did not improve over Phase 6)
# patience=20, hsv_h=0.5, hsv_s=0.5, hsv_v=0.4,
# fliplr=0.5, flipud=0.3, mosaic=1.0, mixup=0.2

print("YOLOv8 training complete!")
print(f"Best model saved at: {results.save_dir}")

#Step 26: Phase 5 — Collect YOLO auto-generated metrics
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

# Step 26 Phase 6: Collect YOLO auto-generated metrics
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

#Step 26: Phase 7 just change the top two lines to the below
# For Phase 7 change to:
output_dir  = '/kaggle/working/phase7_metrics'
zip_path    = '/kaggle/working/phase7_metrics.zip'
phase_label = 'phase7'

#STAGE 12: Phase 5: Two-Stage Inference Pipeline + Annotated Image Generation
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


