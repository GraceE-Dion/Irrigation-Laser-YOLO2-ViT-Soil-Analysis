"""
Script 04: ViT training — Phase 1 (regularized) and Phase 2 (augmented)
Baseline whole-image classification on full soil moisture images
Phase 1 best result: 96.5% validation accuracy
Phase 2 best result: 94.58% validation accuracy
Soil Moisture Classification — ViT + YOLOv8 Pipeline
"""

import os
import numpy as np
import evaluate
import torch
from datasets import load_dataset
from datasets import Image as HFImage
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torchvision import transforms
from PIL import Image as PILImage

MASTER_DIR = '/kaggle/working/Master_Soil_Moisture'

# ── Load dataset ──────────────────────────────────────────────────────────
raw_ds = load_dataset("imagefolder", data_dir=MASTER_DIR, drop_labels=False)
raw_ds = raw_ds.cast_column("image", HFImage(decode=True))

# Build HuggingFace index correction map
folders       = sorted(os.listdir(os.path.join(MASTER_DIR, 'train')))
hf_to_correct = {idx: int(folder) for idx, folder in enumerate(folders)}

def remap_label(example):
    example['label'] = hf_to_correct[example['label']]
    return example

raw_ds = raw_ds.map(remap_label)
print("Labels remapped. Dataset:", raw_ds)

# ── Processor ─────────────────────────────────────────────────────────────
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# ── Augmentation pipeline (Phase 2) ───────────────────────────────────────
train_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    # Processor handles ToTensor and Normalize
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

prepared_ds_train = raw_ds['train'].with_transform(transform_train)
prepared_ds_val   = raw_ds['validation'].with_transform(transform_val)
prepared_ds_test  = raw_ds['test'].with_transform(transform_val)

# ── Model ─────────────────────────────────────────────────────────────────
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=11,
    id2label={i: f"Level {i}" for i in range(11)},
    label2id={f"Level {i}": i for i in range(11)},
    ignore_mismatched_sizes=True,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)

# ── Metrics ───────────────────────────────────────────────────────────────
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# ── Training arguments (Phase 2 — augmented, 25 epochs) ───────────────────
# For Phase 1 (regularized only): set num_train_epochs=17, remove augmentation
training_args = TrainingArguments(
    output_dir="./results_v3",
    save_total_limit=1,
    save_strategy="no",
    load_best_model_at_end=False,
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_ds_train,
    eval_dataset=prepared_ds_val,
    processing_class=processor,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save model
trainer.save_model('./results_v3/final_model')
processor.save_pretrained('./results_v3/final_model')
print("Model saved to ./results_v3/final_model")
