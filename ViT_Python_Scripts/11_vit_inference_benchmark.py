"""
Script 11: ViT inference speed benchmark
Measures per-image latency for ViT-Base-patch16-224 on Kaggle T4 GPU
Required for comparison with MambaVision inference speed (0.90-0.98 ms/image)
Soil Moisture Classification — ViT + YOLOv8 Pipeline
"""

import os
import time
import torch
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

# ── Load saved Phase 2 model ──────────────────────────────────────────────
MODEL_PATH = './results_v3/final_model'

processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
model     = ViTForImageClassification.from_pretrained(MODEL_PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = model.to(device)
model.eval()

print(f"Model loaded on: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Load sample test images ───────────────────────────────────────────────
MASTER_DIR = '/kaggle/working/Master_Soil_Moisture'
test_images = []

for cls_folder in os.listdir(os.path.join(MASTER_DIR, 'test')):
    cls_path = os.path.join(MASTER_DIR, 'test', cls_folder)
    for img_file in os.listdir(cls_path)[:5]:  # 5 images per class
        img_path = os.path.join(cls_path, img_file)
        try:
            img = Image.open(img_path).convert("RGB")
            test_images.append(img)
        except Exception:
            continue
    if len(test_images) >= 100:
        break

print(f"Loaded {len(test_images)} test images for benchmarking.")

# ── Warmup (10 passes) ────────────────────────────────────────────────────
print("Warming up...")
with torch.no_grad():
    for img in test_images[:10]:
        inputs = processor(images=img, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _ = model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

# ── Benchmark — single image, batch size 1 ───────────────────────────────
print("Benchmarking single-image inference (batch size 1)...")
latencies = []

with torch.no_grad():
    for img in test_images:
        inputs = processor(images=img, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        _ = model(**inputs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        latencies.append((t_end - t_start) * 1000)  # convert to ms

latencies = np.array(latencies)

print("\n=== ViT-Base Inference Latency Benchmark ===")
print(f"Device:          {device}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
print(f"Images tested:   {len(latencies)}")
print(f"Mean latency:    {latencies.mean():.2f} ms/image")
print(f"Std deviation:   {latencies.std():.2f} ms")
print(f"Median latency:  {np.median(latencies):.2f} ms/image")
print(f"Min latency:     {latencies.min():.2f} ms/image")
print(f"Max latency:     {latencies.max():.2f} ms/image")
print(f"P95 latency:     {np.percentile(latencies, 95):.2f} ms/image")

# ── Comparison table ──────────────────────────────────────────────────────
vit_mean = latencies.mean()
mamba_crop_latency  = 0.90  # ms/image — from MambaVision benchmark (RTX 3090)
mamba_full_latency  = 0.98  # ms/image — from MambaVision benchmark (RTX 3090)

print("\n=== Architecture Latency Comparison ===")
print(f"{'Model':<35} {'Latency (ms/img)':<20} {'Hardware'}")
print("-" * 70)
print(f"{'ViT-Base-patch16-224':<35} {vit_mean:<20.2f} {'Kaggle T4'}")
print(f"{'MambaVision_S (laser crops)':<35} {mamba_crop_latency:<20.2f} {'MTSU RTX 3090'}")
print(f"{'MambaVision_S (full image)':<35} {mamba_full_latency:<20.2f} {'MTSU RTX 3090'}")
print("\nNote: ViT and MambaVision run on different hardware.")
print("Direct latency comparison should account for T4 vs RTX 3090 difference.")
print("For hardware-normalized comparison, use parameter count and FLOPs.")

# ── Parameter count comparison ────────────────────────────────────────────
vit_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"\n=== Parameter Count ===")
print(f"ViT-Base:        {vit_params:.1f}M parameters")
print(f"MambaVision_S:   ~50.0M parameters")
print(f"Reduction:       {100*(1 - 50/vit_params):.1f}% fewer parameters in MambaVision_S")
