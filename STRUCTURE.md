# 📂 Project File Structure

This repository is organized into a 6-stage functional pipeline, allowing for modular testing and deployment of the Soil Moisture ViT model.

---

🛠️ The 6-Stage Workflow
Each file represents a critical step in the Machine Learning lifecycle:

1. **`01_setup_environment.py`** *Configures GPU T4 x2 acceleration and installs Hugging Face Transformers.*

2. **`02_data_acquisition.py`** *Automated API retrieval for 7 Roboflow datasets (IR, UV, and Standard spectrums).*

3. **`03_automated_merge.py`** *The "Data Fusion" logic. Synchronizes different light spectrums into a unified dataset.*

4. **`04_preprocessing.py`** *Applies ViT Image Processing to normalize laser reflection textures.*

5. **`05_vit_finetuning.py`** *Fine-tuning the Vision Transformer (ViT) on 11 moisture classes (0-10).*

6. **`06_final_evaluation.py`** *Generates the final performance metrics and Confusion Matrix (98% Accuracy).*

---

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

🚀 Production & Documentation

1. master_training_script.py The full, end-to-end pipeline. Run this file to replicate the entire project in one click.

2. README.md The main project report, performance summary, and visual proof.

