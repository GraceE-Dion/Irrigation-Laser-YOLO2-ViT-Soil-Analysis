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
| `09_phase4a_noise_augmentation.py` | Phase 4A | Physical noise augmentation tripling training set — 89.66% |
| `10_phase4b_weighted_loss.py` | Phase 4B | Inverse frequency class weighting targeting weak classes — 90.64% |
| `11_phase5_yolo_detection.py` | Phase 5 | YOLOv8 single-pass laser detection and classification — 95.5% mAP50 |
| `12_phase6_corrected_annotations.py` | Phase 6 | Corrected class mapping + HuggingFace index fix — 95.3% mAP50, 89.1% inference accuracy |
| `13_phase7_augmentation.py` | Phase 7 | Targeted augmentation for IR laser performance — 93.7% mAP50 (negative finding) |
| `14_inference_pipeline.py` | Inference | Two-stage inference with annotated output images and bounding boxes |
| `master_training_script.py` | Full Pipeline | End-to-end pipeline from data download to final inference |
| `requirements.txt` | Setup | All dependencies for full reproduction |



🌳 Repository Tree
```text
Irrigation-Laser-YOLO2-ViT-Soil-Analysis/
├── images/                                 # README metrics and graphs
├── data/
│   ├── training-data/downloaded/           # All 7 Roboflow datasets
│   ├── Master_Laser_Crops/                 # Cropped laser region dataset
│   ├── Master_Soil_Moisture/               # Consolidated dataset with corrected class folders
│   └── Master_YOLO/                        # YOLO-formatted dataset with remapped labels
├── runs/
│   ├── train/merged/                       # Results from multi-spectrum merge
│   ├── fine-tuned/                         # ViT training outputs (all phases)
│   └── yolo_results/                       # YOLOv8 training outputs and metrics
├── 01_setup_environment.py                 # GPU config and library install
├── 02_data_acquisition.py                  # Automated Roboflow API retrieval
├── 03_automated_merge.py                   # Multi-spectrum data fusion
├── 04_preprocessing.py                     # ViT image normalization
├── 05_baseline_training.py                 # Original ViT baseline (overfit)
├── 06_phase1_overfitting_fix.py            # Regularization and early stopping
├── 07_phase2_augmentation.py               # On-the-fly augmentation
├── 08_phase3_laser_crop.py                 # Laser region isolation
├── 09_phase4a_noise_augmentation.py        # Physical noise augmentation
├── 10_phase4b_weighted_loss.py             # Class-weighted loss function
├── 11_phase5_yolo_detection.py             # YOLOv8 object detection
├── 12_phase6_corrected_annotations.py      # Class mapping fix + HuggingFace index correction
├── 13_phase7_augmentation.py               # Targeted IR augmentation (negative finding)
├── 14_inference_pipeline.py                # Final inference and annotation
├── master_training_script.py               # Full end-to-end pipeline
├── requirements.txt                        # All dependencies
└── README.md                               # Main project report
```

---

### 🚀 Production and Documentation

**master_training_script.py**
The full end-to-end pipeline. Run this file to replicate the entire 
five-phase development from data download to final YOLOv8 inference 
in one execution.

**requirements.txt**
Install all dependencies with:

**README.md**
The main project report covering all five development phases, 
performance metrics, convergence analysis, and visual evidence.
