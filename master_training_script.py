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
