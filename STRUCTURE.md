# 📂 Project File Structure

This repository is organized into a modular pipeline covering all five development 
phases, from baseline ViT training through to YOLOv8 object detection.

### The Complete Workflow

| File | Stage | Description |
|---|---|---|
| `01_setup_environment.py` | Setup | Configures GPU T4 x2 acceleration and installs all dependencies |
| `02_data_acquisition.py` | Data | Automated API retrieval for 7 Roboflow datasets (IR, UV, Standard) |
| `03_automated_merge.py` | Data | Synchronizes different light spectrums into a unified dataset |
| `04_preprocessing.py` | Data | Applies ViT Image Processing to normalize laser reflection textures |
| `05_baseline_training.py` | Baseline | Original whole-image ViT training — 10 epochs, 98.11% (overfit) |
| `06_phase1_overfitting_fix.py` | Phase 1 | Regularized ViT with dropout, weight decay, early stopping — 96.5% |
| `07_phase2_augmentation.py` | Phase 2 | On-the-fly augmentation on whole images — 94.58% |
| `08_phase3_laser_crop.py` | Phase 3 | Laser region isolation using bounding box coordinates — 87.68% |
| `09_phase4a_noise_augmentation.py` | Phase 4A | Physical noise augmentation tripling training set to 2,151 images — 89.66% |
| `10_phase4b_weighted_loss.py` | Phase 4B | Inverse frequency class weighting targeting weak classes — 90.64% |
| `11_phase5_yolo_detection.py` | Phase 5 | YOLOv8 single-pass laser detection and moisture classification — 95.5% mAP50 |
| `12_inference_pipeline.py` | Inference | Two-stage inference with annotated output images and bounding boxes |
| `master_training_script.py` | Full Pipeline | End-to-end pipeline from data download to final inference |
| `requirements.txt` | Setup | All dependencies for full reproduction |



🌳 Repository Tree
```text
Irrigation-Laser-Yolo2/
├── data/training-data/downloaded/  # All 7 Roboflow datasets
├── runs/train/merged/              # Results from multi-spectrum merge
├── runs/fine-tuned/                # Final ViT training outputs
├── 01_setup_environment.py         # GPU config & library install
├── 02_data_acquisition.py          # Automated API retrieval
├── 03_automated_merge.py           # Multi-spectrum data fusion
├── 04_preprocessing.py             # ViT image normalization
├── 05_vit_finetuning.py            # Vision Transformer training
├── 06_final_evaluation.py          # Confusion Matrix generation
├── master_training_script.py       # Full end-to-end pipeline
└── README.md                       # Main project report

---

### 🚀 Production and Documentation

**`master_training_script.py`**
The full end-to-end pipeline. Run this file to replicate the entire 
five-phase development from data download to final YOLOv8 inference 
in one execution.

**`requirements.txt`**
Install all dependencies with:

**`README.md`**
The main project report covering all five development phases, 
performance metrics, convergence analysis, and visual evidence.
