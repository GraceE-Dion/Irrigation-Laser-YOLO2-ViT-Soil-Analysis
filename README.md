# Irrigation-Laser Multi-Spectral Soil Moisture Classification via Vision Transformer and YOLOv8 Object Detection

## **Project Summary**

This project represents a shift from manual, error-prone data handling to a Fully Automated Unified Fine-Tuning Pipeline. By transitioning from standard Convolutional Neural Networks (CNNs) to a Vision Transformer (ViT) architecture, this system classifies soil moisture levels (0 - 10) with 98% accuracy.
The core innovation lies in the automated synchronization of 7 disparate datasets, including Infrared (IR), Ultraviolet (UV), and Standard Spectrum, to create a robust model that identifies soil moisture signatures invisible to the human eye.

Beyond model performance, this project integrates AI governance principles into every phase of development, including structured validation frameworks, dataset integrity controls, bias identification and mitigation, deployment risk assessment, and explainability mechanisms. The work is designed to support the development of trustworthy, auditable AI systems aligned with responsible AI frameworks, including NIST's AI Risk Management Framework (AI RMF 1.0) and Executive Order 14110 on the Safe, Secure, and Trustworthy Development and Use of Artificial Intelligence.

### **The Logic**

Instead of simply matching colors, the model analyzes the physical interaction between laser light and soil surfaces:

•	**Specular vs. Diffuse Scattering**: Traditional models are often "distracted" by soil color. Our ViT focuses on micro-textures. Wet soil acts more like a mirror (specular reflection), while dry soil scatters laser light in a rough, noisy pattern (diffuse scattering).

•	**Multi-Spectral Signatures**: By fusing IR and UV data, the model detects thermal and mineral "fingerprints" of water. This ensures the sensor remains accurate regardless of lighting conditions or soil types.

•	**Attention-Based Analysis**: Unlike CNNs that look at small pixel clusters, the Self-Attention mechanism in the ViT evaluates the entire laser spread simultaneously. This allows the model to understand the relationship between different parts of the light spread, leading to higher precision.

---

## 📚 Roboflow Source Datasets
The following 7 specialized datasets were synchronized and merged into the unified Vision Transformer (ViT) training pipeline:

| Dataset ID (Roboflow) | Spectrum / Category | Focus Area |
| :--- | :--- | :--- |
| `soil-moisture-v4` | **Standard Visible** | Baseline laser reflection patterns. |
| `soil-moisture-v4-ir` | **Infrared (IR)** | Thermal signatures of moisture levels. |
| `soil-moisture-v4-uv` | **Ultraviolet (UV)** | High-contrast mineral/moisture separation. |
| `soil-moisture-ir` | **Infrared (IR)** | Secondary heat-based validation. |
| `soil-moisture-5sagf` | **General Field** | Diverse environmental testing conditions. |
| `soil_moisture_september` | **Temporal (Sept)** | Seasonal moisture variations (Standard). |
| `soil_moisture_stir_september`| **Temporal (Sept)** | Specialized "Stirred Soil" reflectance. |

**TOTAL UNIFIED DATA: Multi-Spectrum Data fused into 11 Moisture Soil Levels (0-10)**
---


## Data Processing & Methodology: The 6-Stage Pipeline

To ensure the Vision Transformer could accurately generalize soil moisture levels from complex spectral signatures, the project followed a rigorous 6-stage development pipeline:

1. **Multi-Source Data Acquisition** Raw spectral data was captured simultaneously across three distinct light bands: Infrared (IR), Ultraviolet (UV), and Standard RGB. This triple-source approach ensures that the model receives a holistic view of the laser-soil interaction, capturing thermal and chemical signatures invisible to standard sensors.

2. **Automated Consolidation and Mapping** The asynchronous data streams were synchronized and mapped into a unified tensor format. This stage involved precise spatial alignment (image registration) to ensure that a "patch" in the IR frame corresponded exactly to the same physical coordinate in the UV and RGB frames.

3. **Feature Extraction** Utilizing the Vision Transformer’s patch-based embedding, the model decomposed the consolidated images into 16x16 flattened vector projections. This allowed the architecture to identify high-dimensional features, such as laser refraction intensity and moisture-dependent light scattering.

4. **Model Specialization** The ViT-Base architecture was specialized for this task by modifying the MLP head to classify 11 distinct moisture levels (0–10). This involved fine-tuning the transformer layers to prioritize the spectral fusion of the three input sources.

5. **Training & Real-Time Evaluation** The model underwent 10 epochs of supervised learning on Dual T4 GPUs. Real-time evaluation was performed at the end of each epoch using a partitioned validation set to monitor convergence and prevent overfitting, ensuring the "Attention" weights were stabilizing correctly.

6. **Visualization Results** Final outputs were processed through a Confusion Matrix and Classification Report to visualize model precision. This stage confirmed the model's ability to distinguish between nearly identical moisture levels with 98.11% accuracy.

---

## 🚀 Performance Results

The model development progressed through five systematic phases, each addressing specific limitations identified in the previous stage. The table below summarizes the performance trajectory from the baseline overfit model to the final YOLOv8 object detection architecture.

### Overall Phase Comparison

| Stage | Approach | Accuracy | Status |
|---|---|---|---|
| Baseline | Whole image ViT, no regularization | 98.11% | Overfit — inflated |
| Phase 1 | Whole image ViT, regularized | 96.5% | Honest baseline |
| Phase 2 | Whole image ViT, augmented | 94.58% | Stable |
| Phase 3 | Laser crop ViT, 40 epochs | 87.68% | Architecture change |
| Phase 4A | Laser crop + noise augmentation | 89.66% | Improving |
| Phase 4B | Laser crop + weighted loss | 90.64% | Best ViT result |
| Phase 5 | YOLOv8 object detection | 95.5% mAP50 / 81.25% inference | Superseded — incomplete dataset |
| **Phase 6** | **YOLOv8 + corrected annotations + class mapping fix** | **95.3% mAP50 / 89.1% inference** | **Current best — complete dataset** |
| Phase 7 | YOLOv8 + targeted augmentation | 93.7% mAP50 | Negative finding |

### Phase 1 Key Metrics
- **Validation Accuracy:** 96.5%
- **Epochs:** 17 (early stopping triggered)
- **Key fix:** Dropout regularization, cosine LR scheduling, weight decay

### Phase 2 Key Metrics
- **Validation Accuracy:** 94.58%
- **Epochs:** 25
- **Key fix:** On-the-fly augmentation applied to training data only

### Phase 3 Key Metrics
- **Validation Accuracy:** 87.68%
- **Epochs:** 40
- **Key fix:** Laser region cropped using bounding box coordinates before classification

### Phase 4A Key Metrics
- **Validation Accuracy:** 89.66%
- **Training images:** 2,151 (tripled from 717)
- **Epochs:** 40
- **Key fix:** Gaussian and salt-and-pepper noise copies physically saved to disk

### Phase 4B Key Metrics
- **Validation Accuracy:** 90.64%
- **Epochs:** 40
- **Key fix:** Inverse frequency class weighting targeting Levels 2, 4, and 6

### Phase 5 Key Metrics
- **Overall mAP50:** 95.5%
- **Epochs:** 46 (early stopping at patience=10, best at epoch 36)
- **Architecture:** YOLOv8s — detects laser spot and predicts moisture level in single forward pass
- **Inference accuracy:** 81.25% across 48 unseen images (92.7% excluding soil_moisture_september dataset)

 ### Phase 6 Key Metrics
- **Overall mAP50:** 95.3%
- **Precision:** 0.895
- **Recall:** 0.937
- **Best epoch:** 32 / 42 (EarlyStopping patience=10)
- **Training images:** 1,026 (up from 717 in Phase 5)
- **Inference accuracy:** 89.1% across 46 unseen images (41/46 correct)
- **Datasets updated:** soil_moisture_september v8, soil_moisture_stir_september v4

**Pipeline fixes implemented in Phase 6:**
- Step 4: Added `Level_X` format mapping entries for Roboflow-renamed classes
- Step 4B: HuggingFace alphabetical index correction map for correct numerical class assignment
- Step 6: Label remapping to correct numerical class indices before ViT training
- Step 12: Added `Level_X` mapping entries for laser crop pipeline
- Step 23: Added `Level_X` mapping entries for YOLO dataset preparation
- Step 27: Fixed `soil-moisture-8.2` ground truth display

**Per-class mAP50 (Phase 6):**

| Class | mAP50 | Class | mAP50 |
|---|---|---|---|
| Level_0 | 92.7% | Level_6 | 94.6% |
| Level_1 | 89.1% | Level_7 | 93.6% |
| Level_2 | 86.5% | Level_8 | 91.1% |
| Level_3 | 89.4% | Level_9 | 70.2% |
| Level_4 | 97.2% | Level_10 | 75.9% |
| Level_5 | 95.4% | **all** | **95.3%** | 

### Phase 7 Key Metrics
- **Overall mAP50:** 93.7%
- **Precision:** 0.939
- **Recall:** 0.909
- **Best epoch:** 50 / 50 (no early stopping triggered)
- **Training images:** 1,026 (same as Phase 6)
- **Inference accuracy:** 86.9% across 46 unseen images (40/46 correct)
- **Augmentation parameters added:** hsv_h=0.5, hsv_s=0.5, hsv_v=0.4, fliplr=0.5, flipud=0.3, mosaic=1.0, mixup=0.2, patience=20

**Outcome:** Phase 7 did not improve over Phase 6. The aggressive hue shift (hsv_h=0.5) intended to simulate IR laser appearance during training introduced noise that hurt adjacent moisture level discrimination at levels 8, 9, and 10. Phase 6 remains the production model.

### Cross-Dataset Inference Results — Phase 5 vs Phase 6 vs Phase 7

| Dataset | Phase 5 | Phase 6 | Phase 7 | Notes |
|---|---|---|---|---|
| soil-moisture-v4 | 8/8 ✓ | 8/8 ✓ | 8/8 ✓ | Consistently perfect |
| soil-moisture-v4-ir | 7/7 ✓ | 7/7 ✓ | 7/7 ✓ | Consistently perfect |
| soil-moisture-v4-uv | 7/7 ✓ | 7/7 ✓ | 7/7 ✓ | Consistently perfect |
| soil-moisture-ir | 6/7 | 7/7 ✓ | 7/7 ✓ | Improved Phase 6 onwards |
| soil-moisture-5sagf | 7/7 ✓ | 7/7 ✓ | 7/7 ✓ | Consistently perfect |
| soil_moisture_september | 1/7 | 4/7 | 3/6 | Phase 6 best |
| soil_moisture_stir_september | 5/7 | 1/5 | 1/5 | IR limitation unchanged |
| **Total** | **39/48 (81.25%)** | **41/46 (89.1%)** | **40/46 (86.9%)** | **Phase 6 best overall** |

> **Key Finding:** Phase 6 remains the best performing model with 89.1% inference accuracy. Phase 7 targeted augmentation confirmed that hue-shift simulation of IR laser signatures is insufficient to improve `soil_moisture_stir_september` classification — accuracy remained at 1/5 (20%) across both phases. Five of seven datasets achieve perfect inference accuracy consistently across Phase 6 and Phase 7. Visual analysis of the stir_september dataset revealed two compounding limitations: **stirring the soil physically disrupts the laser reflection pattern making classification inherently unreliable, and the uncontrolled field capture environment produces laser spots that are frequently dim, small, or entirely invisible**. This contrasts with `soil-moisture-ir`, which is also an IR laser dataset but achieves perfect inference accuracy due to a controlled capture environment - consistent container setup, fixed angle, and stable background — confirming the limitation is environmental capture inconsistency and soil disturbance physics, not IR laser imaging inherently.

> **Recommended future work:** Future work should focus on standardizing the stir_september capture environment and collection methodology rather than augmentation-based simulation, a controlled setup with fixed container, consistent laser angle, and stable lighting would address the root cause identified through visual analysis.

---

## AI Governance and Responsible Development Principles
This project was developed with explicit attention to AI governance principles that extend beyond model accuracy. The following governance mechanisms were integrated throughout the development lifecycle:
- Honest performance reporting: The baseline overfitting result (98.11%) was retained in the evaluation record rather than discarded, demonstrating that negative findings have as much evidentiary value as positive ones.
- Dataset integrity and labeling quality: Systematic auditing identified a probable labeling error in the source data where the model's prediction was more consistent with observable visual evidence than the assigned ground truth label, demonstrating that governance-aware validation can surface data quality issues invisible to standard metrics.
- Bias identification and mitigation: Per-class performance was tracked across all eleven moisture levels and all five development phases, producing a complete audit trail of bias identification and targeted mitigation through inverse-frequency class weighting.
- Deployment risk assessment: Inference testing was conducted across all seven source datasets on unseen images, with dataset-specific reliability profiling enabling practitioners to make risk-informed deployment decisions rather than relying on aggregate accuracy alone.
- Explainability: Annotated inference outputs display bounding boxes, confidence scores, ground truth labels, and correct/incorrect indicators for all 48 test images, providing full transparency of model decisions across diverse imaging conditions.

---

<h2 align="center">Development Phase Metrics</h2>

<h3 align="center">Baseline: Original ViT Model (Overfit)</h3>
<p align="center">
  <img src="images/epoch_training_baseline.jpg" width="49%" />
  <img src="images/classification_report_baseline.jpg" width="49%" />
</p>
<p align="center">
  <img src="images/accuracy_graph_baseline.jpg" width="49%" />
  <img src="images/loss_curve_baseline.jpg" width="49%" />
</p>
<p align="center">
  <img src="images/confusion_matrix_baseline.jpg" width="70%" />
</p>
<hr>

<h3 align="center">Phase 1: Overfitting Correction</h3>
<p align="center">
  <img src="images/epoch_training_phase1.jpg" width="49%" />
  <img src="images/classification_report_phase1.jpg" width="49%" />
</p>
<p align="center">
  <img src="images/accuracy_graph_phase1.jpg" width="49%" />
  <img src="images/loss_curve_phase1.jpg" width="49%" />
</p>
<p align="center">
  <img src="images/confusion_matrix_phase1.jpg" width="70%" />
</p>
<hr>

<h3 align="center">Phase 2: Data Augmentation on Whole Images</h3>
<p align="center">
  <img src="images/epoch_training_phase2.jpg" width="49%" />
  <img src="images/classification_report_phase2.jpg" width="49%" />
</p>
<p align="center">
  <img src="images/accuracy_graph_phase2.jpg" width="49%" />
  <img src="images/loss_curve_phase2.jpg" width="49%" />
</p>
<p align="center">
  <img src="images/confusion_matrix_phase2.jpg" width="70%" />
</p>
<hr>

<h3 align="center">Phase 3: Laser Region Isolation</h3>
<p align="center">
  <img src="images/epoch_training_phase3.jpg" width="49%" />
  <img src="images/classification_report_phase3.jpg" width="49%" />
</p>
<p align="center">
  <img src="images/accuracy_graph_phase3.jpg" width="49%" />
  <img src="images/loss_curve_phase3.jpg" width="49%" />
</p>
<p align="center">
  <img src="images/confusion_matrix_phase3.jpg" width="70%" />
</p>
<hr>

<h3 align="center">Phase 4A: Physical Noise Augmentation</h3>
<p align="center">
  <img src="images/epoch_training_phase4a.jpg" width="49%" />
  <img src="images/classification_report_phase4a.jpg" width="49%" />
</p>
<p align="center">
  <img src="images/accuracy_graph_phase4a.jpg" width="49%" />
  <img src="images/loss_curve_phase4a.jpg" width="49%" />
</p>
<p align="center">
  <img src="images/confusion_matrix_phase4a.jpg" width="70%" />
</p>
<hr>

<h3 align="center">Phase 4B: Class-Weighted Loss Function</h3>
<p align="center">
  <img src="images/epoch_training_phase4b.jpg" width="49%" />
  <img src="images/classification_report_phase4b.jpg" width="49%" />
</p>
<p align="center">
  <img src="images/accuracy_graph_phase4b.jpg" width="49%" />
  <img src="images/loss_curve_phase4b.jpg" width="49%" />
</p>
<p align="center">
  <img src="images/confusion_matrix_phase4b.jpg" width="70%" />
</p>
<hr>

<h3 align="center">Phase 5: YOLOv8 Object Detection</h3>
<p align="center">
  <img src="images/training_curves_phase5.png" width="99%" />
</p>
<p align="center">
  <img src="images/confusion_matrix_phase5.png" width="49%" />
  <img src="images/confusion_matrix_phase5_normalized.png" width="49%" />
</p>
<p align="center">
  <img src="images/phase5_validation_batch_pred.jpg" width="70%" />
</p>
<hr>

<h3 align="center">Phase 6: Corrected Annotations + Class Mapping Fix</h3>
<p align="center">
  <img src="images/phase6_results.png" width="99%" />
</p>
<p align="center">
  <img src="images/phase6_confusion_matrix.png" width="49%" />
  <img src="images/phase6_confusion_matrix_normalized.png" width="49%" />
</p>
<p align="center">
  <img src="images/phase6_val_batch0_pred.jpg" width="70%" />
</p>
<hr>

<h3 align="center">Phase 7: Targeted Augmentation for IR Laser Performance</h3>
<p align="center">
  <img src="images/phase7_results.png" width="99%" />
</p>
<p align="center">
  <img src="images/phase7_confusion_matrix.png" width="49%" />
  <img src="images/phase7_confusion_matrix_normalized.png" width="49%" />
</p>
<p align="center">
  <img src="images/phase7_val_batch0_pred.jpg" width="70%" />
</p>
<hr>


**Convergence Analysis:** The baseline model trained for 10 epochs, with validation 
accuracy climbing from 85.20% at Epoch 1 to a plateau of 98.11% at Epoch 10. However, 
subsequent analysis revealed signs of overfitting — training loss diverged from 
validation loss at Epoch 9, indicating inflated performance. Seven development phases 
were implemented to address this, progressively improving model reliability from a 
regularized 96.5% honest baseline (Phase 1) through to a YOLOv8 object detection 
architecture achieving 95.5% mAP50 (Phase 5).

**Observation:** Across all phases, the model demonstrated consistent improvement in 
diagonal density on the confusion matrix. The final YOLOv8 architecture effectively 
detects UV laser spots and classifies moisture levels in a single forward pass, 
producing perfect results across five of seven datasets and confirming the object 
detection approach as the architecturally correct solution for this task.

## 📊 Training Performance & Convergence

The model development progressed through five phases across multiple training runs on 
Dual T4 GPUs, using Cross-Entropy Loss with progressive architectural improvements. 
The table below summarizes the key convergence metrics across all phases:

| Phase | Approach | Epoch 1 Accuracy | Best Accuracy | Epochs Run |
|---|---|---|---|---|
| Baseline | Whole image ViT | 85.20% | 98.11% (overfit) | 10 |
| Phase 1 | Regularized ViT | 27.09% | 96.5% | 17 (early stop) |
| Phase 2 | Augmented ViT | 13.30% | 94.58% | 25 |
| Phase 3 | Laser crop ViT | 16.74% | 87.68% | 40 |
| Phase 4A | + Noise augmentation | 21.67% | 89.66% | 40 |
| Phase 4B | + Weighted loss | 22.16% | 90.64% | 40 |
| **Phase 5** | **YOLOv8 detection** | — | **95.5% mAP50** | **46 (early stop)** |
| **Phase 6** | **YOLOv8 + corrected annotations** | — | **95.3% mAP50** | **42 (early stop)** |

#### Why the Architecture Evolved: From ViT to YOLOv8

The baseline model achieved rapid convergence driven by the Vision Transformer's 
ability to process global context through Multi-Head Self-Attention:

- **Feature Prioritization:** Attention weights allow the model to ignore background 
soil noise and attend specifically to the laser refraction patterns.
- **Spectral Fusion:** The model learns to prioritize Infrared (IR) data in instances 
where standard RGB shadows might obscure moisture levels.

However, systematic analysis revealed that whole-image classification introduced 
irrelevant background signals — soil debris, plant roots, and variable lighting, 
that limited generalization across datasets. This motivated the progressive shift 
toward laser region isolation and ultimately the YOLOv8 object detection approach, 
which detects the UV laser spot and classifies the moisture level simultaneously in 
a single forward pass.

#### 📊 Key Observations Across All Phases:

- **Overfitting Correction:** The baseline 98.11% accuracy was inflated due to 
overfitting. Phase 1 regularization produced a more honest 96.5% baseline with 
training and validation losses converging cleanly throughout 17 epochs.
- **Augmentation Impact:** Physical noise augmentation in Phase 4A tripled the 
training set from 717 to 2,151 images, improving accuracy from 87.68% to 89.66% 
and bringing Level 10 F1 score to a perfect 1.00.
- **Class Imbalance Resolution:** Weighted loss in Phase 4B specifically targeted 
the weakest classes (Levels 2, 4, and 6), achieving the best ViT result of 90.64%.
- **Architectural Breakthrough:** YOLOv8 object detection in Phase 5 achieved 
95.5% mAP50, with perfect performance across five of seven datasets, confirming 
that framing the task as object detection is the architecturally correct approach.
- **Extreme Error Elimination:** No extreme errors (confusing Level 0 with Level 10) 
were observed in Phase 5, making the final model viable for real-world automated 
irrigation applications.
- **Dataset Integrity Fix:** Phase 6 resolved a critical class index misalignment in
the september datasets, correctly incorporating 124 previously excluded images and
increasing the training set to 1,026 images. Overall inference accuracy improved
from 81.25% to 89.1%.


---

## 🧪 Real-World Inference Test (Multi-Source Validation)

To validate the model's reliability across all development phases, inference tests 
were performed on unseen samples from all 7 merged datasets. The final Phase 5 
YOLOv8 model was evaluated on 48 unseen images, producing annotated outputs showing 
bounding boxes around detected UV laser spots with simultaneous moisture level 
predictions and ground truth labels for direct comparison.

The inference results confirm that the YOLOv8 architecture generalizes effectively 
across RGB, IR, and UV spectral modalities. Of the 9 mismatches observed across 
48 inference images, 6 originated from the soil_moisture_september dataset, a 
dataset with known annotation limitations where bounding boxes cover the full image 
area, providing no meaningful laser localization for the detection model. Excluding 
this dataset, Phase 5 achieves 92.7% inference accuracy across the remaining six 
well-annotated datasets.

To ensure total reproducibility and data integrity, the following mapping log was 
generated during the validation session between the generic labels used in this 
documentation and the unique Roboflow file hashes present in the dynamic training 
environment.

### 📊 Detailed Inference Output — Phase 6 (Best Model)

<h3 align="center">Multi-spectral ViT: Test Set Input Samples</h3>

<h4 align="center">Dataset 1: Soil_Moisture_V4</h4>
<p align="center">
  <img src="images/sample_01_soil-moisture-v4.jpg" width="24%" />
  <img src="images/sample_02_soil-moisture-v4.jpg" width="24%" />
  <img src="images/sample_03_soil-moisture-v4.jpg" width="24%" />
  <img src="images/sample_04_soil-moisture-v4.jpg" width="24%" />
</p>
<p align="center">
  <img src="images/sample_05_soil-moisture-v4.jpg" width="24%" />
  <img src="images/sample_06_soil-moisture-v4.jpg" width="24%" />
  <img src="images/sample_07_soil-moisture-v4.jpg" width="24%" />
  <img src="images/sample_08_soil-moisture-v4.jpg" width="24%" />
</p>

<hr>

<h4 align="center">Dataset 2: Soil_Moisture_V4_IR</h4>
<p align="center">
  <img src="images/sample_09_soil-moisture-v4-ir.jpg" width="24%" />
  <img src="images/sample_10_soil-moisture-v4-ir.jpg" width="24%" />
  <img src="images/sample_11_soil-moisture-v4-ir.jpg" width="24%" />
  <img src="images/sample_12_soil-moisture-v4-ir.jpg" width="24%" />
</p>
<p align="center">
  <img src="images/sample_13_soil-moisture-v4-ir.jpg" width="24%" />
  <img src="images/sample_14_soil-moisture-v4-ir.jpg" width="24%" />
  <img src="images/sample_15_soil-moisture-v4-ir.jpg" width="24%" />
</p>

<hr>

<h4 align="center">Dataset 3: Soil_Moisture_V4_UV</h4>
<p align="center">
  <img src="images/sample_16_soil-moisture-v4-uv.jpg" width="24%" />
  <img src="images/sample_17_soil-moisture-v4-uv.jpg" width="24%" />
  <img src="images/sample_18_soil-moisture-v4-uv.jpg" width="24%" />
  <img src="images/sample_19_soil-moisture-v4-uv.jpg" width="24%" />
</p>
<p align="center">
  <img src="images/sample_20_soil-moisture-v4-uv.jpg" width="24%" />
  <img src="images/sample_21_soil-moisture-v4-uv.jpg" width="24%" />
  <img src="images/sample_22_soil-moisture-v4-uv.jpg" width="24%" />
</p>

<hr>

<h4 align="center">Dataset 4: Soil_Moisture_IR</h4>
<p align="center">
  <img src="images/sample_23_soil-moisture-ir.jpg" width="24%" />
  <img src="images/sample_24_soil-moisture-ir.jpg" width="24%" />
  <img src="images/sample_25_soil-moisture-ir.jpg" width="24%" />
  <img src="images/sample_26_soil-moisture-ir.jpg" width="24%" />
</p>
<p align="center">
  <img src="images/sample_27_soil-moisture-ir.jpg" width="24%" />
  <img src="images/sample_28_soil-moisture-ir.jpg" width="24%" />
  <img src="images/sample_29_soil-moisture-ir.jpg" width="24%" />
</p>

<hr>

<h4 align="center">Dataset 5: Soil_Moisture_5SAGF</h4>
<p align="center">
  <img src="images/sample_30_soil-moisture-5sagf.jpg" width="24%" />
  <img src="images/sample_31_soil-moisture-5sagf.jpg" width="24%" />
  <img src="images/sample_32_soil-moisture-5sagf.jpg" width="24%" />
  <img src="images/sample_33_soil-moisture-5sagf.jpg" width="24%" />
</p>
<p align="center">
  <img src="images/sample_34_soil-moisture-5sagf.jpg" width="24%" />
  <img src="images/sample_35_soil-moisture-5sagf.jpg" width="24%" />
  <img src="images/sample_36_soil-moisture-5sagf.jpg" width="24%" />
</p>

<hr>

<h4 align="center">Dataset 6: Soil_Moisture_September</h4>
<p align="center">
  <img src="images/sample_37_soil_moisture_september.jpg" width="24%" />
  <img src="images/sample_38_soil_moisture_september.jpg" width="24%" />
  <img src="images/sample_39_soil_moisture_september.jpg" width="24%" />
  <img src="images/sample_40_soil_moisture_september.jpg" width="24%" />
</p>
<p align="center">
  <img src="images/sample_41_soil_moisture_september.jpg" width="24%" />
  <img src="images/sample_42_soil_moisture_september.jpg" width="24%" />
  <img src="images/sample_43_soil_moisture_september.jpg" width="24%" />
</p>

<hr>

<h4 align="center">Dataset 7: Soil_Moisture_Stir_September</h4>
<p align="center">
  <img src="images/sample_44_soil_moisture_stir_september.jpg" width="24%" />
  <img src="images/sample_45_soil_moisture_stir_september.jpg" width="24%" />
  <img src="images/sample_46_soil_moisture_stir_september.jpg" width="24%" />
</p>
<p align="center">
  <img src="images/sample_47_soil_moisture_stir_september.jpg" width="24%" />
  <img src="images/sample_48_soil_moisture_stir_september.jpg" width="24%" />
</p>

<p align="center">
 <em>Figure 1: Representative soil samples across 7 multi-spectral datasets used for final model inference (Phase 6 -  best performing model, 89.1% inference accuracy).</em>
</p>

<hr>

<h3 align="center">Detailed Inference Output</h3>


### 🔬 Laser Pattern Visual Investigation

A systematic visual comparison was conducted across four datasets to analyze laser pattern consistency and identify the root cause of `soil_moisture_stir_september` performance limitations. Full investigation code is available in `15_laser_pattern_investigation.py`.

**Visual findings:**

| Dataset | Laser Type | Capture Environment | Laser Visibility | Inference Accuracy |
|---|---|---|---|---|
| soil_moisture_stir_september | IR | Uncontrolled field | Dim, small, or invisible | 20% |
| soil_moisture_september | UV | Field | Consistent blue glow | 57% |
| soil-moisture-v4-uv | UV | Controlled container | Strong, consistent signal | 100% |
| soil-moisture-ir | IR | Controlled container | Consistent white spot | 100% |

**Key conclusion:** Performance is driven by capture environment consistency, not laser wavelength. Both IR datasets perform differently — `soil-moisture-ir` achieves perfect accuracy in a controlled setup while `soil_moisture_stir_september` achieves only 20% in an uncontrolled field environment with disturbed soil surface.

<p align="center">
  <img src="images/laser_pattern_comparison.jpg" width="99%" />
</p>



### **Known Dataset Limitations** 

During inference testing across 48 sampled images drawn from seven Roboflow datasets, the model identified a case in the soil_moisture_stir_september dataset where the predicted label (Level 2) diverged significantly from the assigned ground truth label (Level 10). Visual inspection of the flagged image revealed dry-textured soil with a compact, un-saturated surface, no visible moisture sheen, and a small, concentrated UV laser spot showing minimal diffusion across the soil surface. Diffusion breadth and intensity of the UV laser signal are primary visual indicators of moisture saturation in laser-based detection; the image in question displayed neither characteristic consistent with a Level 10 classification.
This discrepancy suggests a probable labeling error in the source dataset rather than a model failure. It is notable that the ViT classifier, trained on multi-spectral image data across IR, UV, and RGB modalities, produced a prediction more consistent with the observable visual evidence than the assigned ground truth label. This outcome demonstrates that the model learned meaningful moisture-to-visual feature relationships with sufficient fidelity to surface questionable annotations in the training corpus.

This finding highlights a known challenge in laser-based soil moisture classification datasets: UV laser intensity and spatial diffusion patterns can be ambiguous under certain lighting and soil composition conditions, increasing the risk of labeling inconsistencies during manual annotation. Researchers and practitioners applying or extending this model should be aware that a subset of labels in the underlying datasets may not fully reflect actual moisture conditions, particularly in the soil_moisture_stir_september and soil_moisture_september collections where stir-based soil disturbance may have altered surface appearance at the time of capture.

This finding reflects the governance principle that model validation should be designed to surface data quality issues, not merely confirm performance benchmarks. The documentation of this labeling inconsistency in the project repository ensures transparency and supports downstream auditability for researchers extending this work.

**Phase 6 Visual Analysis - Laser Pattern Investigation:** Systematic visual comparison across all seven datasets revealed that performance differences are primarily driven by capture environment consistency rather than laser wavelength. The `soil_moisture_stir_september` dataset shows two compounding issues: (1) stirring the soil physically disrupts the laser reflection pattern, producing irregular surface textures that scatter laser light unpredictably, and (2) the uncontrolled field capture environment results in a laser spot that is frequently dim, small, or entirely invisible in some images, providing insufficient signal for reliable detection. This contrasts with `soil-moisture-ir`, which is also an IR laser dataset but achieves perfect inference accuracy due to a controlled capture environment - consistent container setup, fixed angle, and stable background. This confirms that the classification limitation in stir_september is environmental capture inconsistency compounded by soil disturbance physics, not IR laser imaging inherently. The `soil_moisture_september` dataset improved from 14.7% (Phase 5) to 57% (Phase 6) following annotation correction, confirming that earlier errors were pipeline-related rather than fundamental. These findings reflect the governance principle that inference testing across diverse capture conditions surfaces deployment constraints invisible to aggregate accuracy metrics, enabling practitioners to make risk-informed decisions about real-world deployment contexts.

**Why `soil_moisture_stir_september` accuracy dropped from 71% (Phase 5) to 20% (Phase 6):** This drop is explained by two compounding factors. First, in Phase 5 the class labels for the september datasets were incorrectly assigned due to a class index misalignment - the model was inadvertently receiving credit for correct predictions against misassigned ground truth labels, making the 71% result artificially inflated rather than genuine. Once the annotation errors were corrected in Phase 6 the true model performance was revealed at 20%. Second, the underlying reason genuine performance is low is the fundamental laser reflection issue identified through visual analysis - stirring the soil disrupts the consistent laser diffusion pattern the model relies on, and the uncontrolled field capture environment produces laser spots that are frequently dim, small, or entirely invisible. The 71% Phase 5 result was therefore a false positive masked by annotation errors rather than evidence of genuine model capability on this dataset.

### **Development Phases: From Baseline to Object Detection**

Following the initial baseline model, the research progressed through six systematic improvement phases, each building on findings from the previous stage.

**Phase 1: Overfitting Correction**
The baseline model exhibited overfitting, training loss diverged from validation loss at epoch 9 and validation accuracy plateaued at 97%, indicating inflated performance. Phase 1 introduced dropout regularization (0.1), reduced learning rate (2e-5), weight decay (0.01), cosine learning rate scheduling, and early stopping with patience of 3 epochs. The model trained for 17 epochs before early stopping triggered, achieving an honest baseline of 96.5% validation accuracy with significantly reduced overfitting.

**Phase 2: Data Augmentation on Whole Images**
Phase 2 applied on-the-fly augmentation to the training pipeline including random flipping, color jitter, random resized crop, and Gaussian blur, applied exclusively to training data. The model trained for 25 epochs and achieved 94.58% validation accuracy. While augmentation stabilized training further, overall accuracy did not improve beyond Phase 1, confirming that augmenting whole images containing irrelevant background content had limited impact on the laser-specific classification signal.

**Phase 3: Laser Region Isolation**
Following instructor guidance, Phase 3 introduced a two-stage pipeline using bounding box coordinates from the original YOLOv5 dataset labels to crop the UV laser region from each image before classification, with 5% padding to preserve edge diffusion patterns. The ViT was retrained on cropped laser regions only for 40 epochs, achieving 87.68% validation accuracy. The loss curves showed the cleanest convergence of all ViT phases. However, mid-range moisture levels (3, 4, 6) showed increased confusion due to inconsistent crop sizes across datasets, with laser spots ranging from 6% to 88% of image area causing significant upscaling inconsistency.

**Phase 4A: Physical Noise Augmentation on Laser Crops**
Phase 4A physically generated augmented training images by saving Gaussian noise and salt-and-pepper noise copies of each training image to disk, effectively tripling the training set from 717 to 2,151 images. This approach directly followed instructor guidance to add noise to images and train on the combined original and augmented data together. The model trained for 40 epochs achieving 89.66% validation accuracy, with Level 10 improving to perfect 1.00 F1 score.

**Phase 4B: Class-Weighted Loss Function**
Phase 4B added inverse frequency class weighting to the loss function, specifically targeting the weakest performing classes. A custom WeightedTrainer was implemented using PyTorch CrossEntropyLoss with per-class weights computed from training sample counts. Training for 40 epochs on the augmented dataset achieved 90.64% validation accuracy, the best ViT result across all phases.

**Phase 5: YOLOv8 Object Detection (Final Architecture)**
To frame the problem as an object detection task, Phase 5 trained a YOLOv8s model treating each moisture level (0-10) as a distinct object class. YOLOv8 detects the UV laser spot and predicts the moisture level simultaneously in a single forward pass, eliminating the two-stage pipeline entirely. The model trained for 46 epochs before early stopping triggered, achieving 95.5% mAP50 across all 11 classes, a significant improvement over all ViT phases.

**Phase 6: Corrected Annotations + Class Mapping Fix**
Phase 6 addressed a critical data pipeline issue discovered during inference testing, a class index misalignment caused by Roboflow exporting class names in alphabetical rather than numerical order, causing `Level_10` to be assigned index 2 instead of index 10 across the september datasets. This resulted in 124 training images being incorrectly labeled or excluded entirely. Phase 6 introduced a HuggingFace label remapping correction (Step 4B), updated class mapping dictionaries across Steps 4, 12, and 23, and fixed a ground truth display error for the `soil-moisture-8.2` class. The corrected pipeline increased the training dataset from 717 to 1,026 images and improved overall inference accuracy from 81.25% to 89.1%, with five of seven datasets achieving perfect inference accuracy. The model trained for 42 epochs before early stopping triggered, achieving 95.3% mAP50 across all 11 classes.

**Phase 7: Targeted Augmentation for IR Laser Performance**
Phase 7 applied aggressive augmentation parameters to address the IR laser classification weakness identified in Phase 6, specifically the `soil_moisture_stir_september` dataset which achieved only 1/5 inference accuracy. Augmentation included large hue shifts (hsv_h=0.5) to simulate IR laser color signatures, saturation and brightness variation (hsv_s=0.5, hsv_v=0.4), horizontal and vertical flipping, mosaic augmentation, and mixup (0.2) to improve adjacent class boundary learning. EarlyStopping patience was increased from 10 to 20 epochs. The model trained for all 50 epochs achieving 93.7% mAP50. However inference testing showed overall accuracy dropped from 89.1% to 86.9%, with `soil_moisture_stir_september` remaining at 1/5 and `soil_moisture_september` worsening from 4/7 to 3/6. The aggressive hue shift introduced training noise that degraded high moisture level discrimination (Levels 8, 9, 10) without meaningfully improving IR laser detection. Phase 6 remains the best performing model. This negative finding confirms that simulating IR laser appearance through hue augmentation alone is insufficient, dedicated IR training data collection remains the recommended path for Phase 8.

---

### Complete Phase Comparison

| Phase | Approach | Best Accuracy/mAP50 |
|---|---|---|
| Original | Whole image ViT, no regularization | 97% (overfit) |
| Phase 1 | Whole image ViT, regularized | 96.5% |
| Phase 2 | Whole image ViT, augmented | 94.58% |
| Phase 3 | Laser crop ViT, 40 epochs | 87.68% |
| Phase 4A | Laser crop + noise augmentation | 89.66% |
| Phase 4B | Laser crop + noise aug + weighted loss | 90.64% |
| **Phase 5** | **YOLOv8 object detection** | **95.5% mAP50** |
| **Phase 6** | **YOLOv8 + corrected annotations + class mapping fix** | **95.3% mAP50** |

---

**Inference Validation**: To confirm real-world viability, the model was tested against unseen samples from all 7 merged datasets, achieving strong classification performance across RGB, IR, and UV spectral modalities. The ViT architecture's self-attention mechanism showed capacity to generalize across diverse soil surface conditions, including stirred and undisturbed scenarios, though edge cases at extreme moisture levels revealed sensitivity to labeling inconsistencies in the source data.

---

## Technical Specification 

| Parameter | ViT Phases | Phase 5 (YOLOv8) | Phase 6 (YOLOv8 corrected) |
|---|---|---|---|
| Architecture | ViT-Base-patch16-224 | YOLOv8s | YOLOv8s |
| Hardware | Dual NVIDIA T4 GPUs | Dual NVIDIA T4 GPUs | NVIDIA T4 GPU |
| Optimizer | AdamW (2e-5 LR) | Adam (0.001 LR) | Adam (0.001 LR) |
| Regularization | Dropout 0.1, weight decay 0.01 | Weight decay 0.0005 | Weight decay 0.0005 |
| Label Smoothing | 0.1 | 0.1 | 0.1 |
| Training Images | 717 (Phase 1-3) / 2,151 (Phase 4A-4B) | 717 | 1,026 |
| Image Size | 224×224 | 640×640 | 640×640 |
| Max Epochs | 40 | 50 (early stop at 46) | 50 (early stop at 42) |
| Best Result | 90.64% accuracy (Phase 4B) | 95.5% mAP50 | 95.3% mAP50 |
| Governance Framework | NIST AI RMF 1.0, EO 14110 | NIST AI RMF 1.0, EO 14110 | NIST AI RMF 1.0, EO 14110 |

The model architecture utilizes a pre-trained ViT-Base backbone. During initialization, the original ImageNet classifier head was replaced with a custom linear layer specialized for 11 soil moisture levels (0–10). This was confirmed by the weight initialization report, ensuring the transformer blocks were fine-tuned specifically to identify spectral diffraction patterns rather than general objects.

---

## 🏁 Conclusion

This project successfully demonstrates that a **Vision Transformer (ViT) architecture** is effective at interpreting complex spectral patterns created by laser-soil interaction. Through five systematic development phases, the research progressed from a baseline whole-image classifier to a **YOLOv8** object detection architecture that simultaneously detects UV laser spots and predicts moisture levels in a single forward pass. The final Phase 6 model achieves 95.3% mAP50 on a fully corrected 1,026-image dataset, with perfect inference accuracy across five of seven datasets and an overall inference accuracy of 89.1%, an improvement of 7.85 percentage points over Phase 5. Phase 7 explored targeted augmentation to improve IR laser dataset performance but confirmed that hue-shift augmentation alone is insufficient to simulate real IR laser signatures. Phase 6 remains the production model with 89.1% inference accuracy across 46 unseen images. Future work will focus on standardizing the stir_september capture environment to address the laser reflection inconsistency identified through visual analysis, and extending the pipeline to real-time field deployment. The integration of multi-spectral data (IR, UV, and RGB) allows for a robust classification system that could significantly improve automated irrigation efficiency and water conservation in precision agriculture. 

Throughout all phases, the project applied governance-first development principles - structured validation, dataset integrity controls, bias mitigation, and deployment risk benchmarking - demonstrating that responsible AI development is achievable within resource-constrained research environments and is essential to producing AI systems that can be trusted, audited, and safely extended across real-world deployment contexts.

