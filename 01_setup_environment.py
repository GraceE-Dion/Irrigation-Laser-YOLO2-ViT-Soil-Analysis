# STAGE 1: Environment Setup
# Installs necessary AI libraries for Vision Transformers and Data Handling
!pip install -q transformers datasets evaluate roboflow
import torch
print(f"GPU Detected: {torch.cuda.is_available()}")
