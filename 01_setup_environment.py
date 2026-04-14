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
