"""
Script 01: Install dependencies and imports
Soil Moisture Classification — ViT + YOLOv8 Pipeline
MTSU / Kaggle T4 GPU
"""

import subprocess
subprocess.run([
    "pip", "install", "-q",
    "evaluate", "roboflow", "datasets", "transformers[torch]", "ultralytics"
], check=True)

import os
import shutil
import yaml
import torch
import numpy as np
import evaluate
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

print("All imports successful.")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
