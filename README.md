# Irrigation-Laser-YOLO2-ViT-Soil-Analysis
## **Project Summary**

This project represents a shift from manual, error-prone data handling to a Fully Automated Unified Fine-Tuning Pipeline. By transitioning from standard Convolutional Neural Networks (CNNs) to a Vision Transformer (ViT) architecture, this system classifies soil moisture levels (0–10) with 98% accuracy.
The core innovation lies in the automated synchronization of 7 disparate datasets, including Infrared (IR), Ultraviolet (UV), and Standard Spectrum, to create a robust model that identifies moisture signatures invisible to the human eye.

### **The Logic**

Instead of simply matching colors, the model analyzes the physical interaction between laser light and soil surfaces:

•	**Specular vs. Diffuse Scattering**: Traditional models are often "distracted" by soil color. Our ViT focuses on micro-textures. Wet soil acts more like a mirror (specular reflection), while dry soil scatters laser light in a rough, noisy pattern (diffuse scattering).

•	**Multi-Spectral Signatures**: By fusing IR and UV data, the model detects thermal and mineral "fingerprints" of water. This ensures the sensor remains accurate regardless of lighting conditions or soil types.

•	**Attention-Based Analysis**: Unlike CNNs that look at small pixel clusters, the Self-Attention mechanism in the ViT evaluates the entire laser spread simultaneously. This allows the model to understand the relationship between different parts of the light spread, leading to higher precision.


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

**TOTAL UNIFIED DATA: Multi-Spectrum Data fused into 11 Moisture Classes (0-10)**
---

## Data Processing & Methodology: The 6-Stage Pipeline

To ensure the Vision Transformer could accurately generalize moisture levels from complex spectral signatures, the project followed a rigorous 6-stage development pipeline:

1. **Multi-Source Data Acquisition** Raw spectral data was captured simultaneously across three distinct light bands: Infrared (IR), Ultraviolet (UV), and Standard RGB. This triple-source approach ensures that the model receives a holistic view of the laser-soil interaction, capturing thermal and chemical signatures invisible to standard sensors.

2. **Automated Consolidation and Mapping** The asynchronous data streams were synchronized and mapped into a unified tensor format. This stage involved precise spatial alignment (image registration) to ensure that a "patch" in the IR frame corresponded exactly to the same physical coordinate in the UV and RGB frames.

3. **Feature Extraction** Utilizing the Vision Transformer’s patch-based embedding, the model decomposed the consolidated images into 16x16 flattened vector projections. This allowed the architecture to identify high-dimensional features, such as laser refraction intensity and moisture-dependent light scattering.

4. **Model Specialization** The ViT-Base architecture was specialized for this task by modifying the MLP head to classify 11 distinct moisture levels (0–10). This involved fine-tuning the transformer layers to prioritize the spectral fusion of the three input sources.

5. **Training & Real-Time Evaluation** The model underwent 10 epochs of supervised learning on Dual T4 GPUs. Real-time evaluation was performed at the end of each epoch using a partitioned validation set to monitor convergence and prevent overfitting, ensuring the "Attention" weights were stabilizing correctly.

6. **Visualization Results** Final outputs were processed through a Confusion Matrix and Classification Report to visualize model precision. This stage confirmed the model's ability to distinguish between nearly identical moisture levels with 98.11% accuracy.


## 🚀 Performance Results
The Vision Transformer model was subjected to a final validation using a hold-out test set from all 7 merged sources.

* **Final Accuracy:** 98.1%
* **Classes:** 11 Moisture Levels (0-10)

 <p align="center">
  <img src="images/training_log.png" width="45%" alt="Training Log" />
  <img src="images/classification_report.png" width="45%" alt="Classification Report" />
</p>

---


## 📊 Training Performance & Convergence
The Vision Transformer (ViT) was subjected to 10 epochs of training using a Cross-Entropy Loss function on Dual T4 GPUs. The model reached stability rapidly:

| Metric | Initial (Epoch 1) | Final (Epoch 10) |
| :--- | :---: | :---: |
| **Validation Loss** | 1.5372 | **0.3695** |
| **Validation Accuracy** | 85.20% | **98.11%** |

#### Why Accuracy is High: Multi-Head Self-Attention
The rapid convergence is driven by the **Vision Transformer's** ability to process global context:
* **Feature Prioritization:** Attention weights allow the model to ignore background soil noise and "attend" specifically to the laser's refraction patterns.
* **Spectral Fusion:** The model learns to prioritize Infrared (IR) data in instances where standard RGB shadows might obscure moisture levels.

#### 📊 Key Observations:
* **Steady Convergence:** A ~76% reduction in loss confirms the model successfully mastered the complex spectral signatures of laser-soil interaction.
* **Class Precision:** The Confusion Matrix shows high diagonal density, meaning the model accurately distinguishes between similar moisture levels (e.g., Level 4 vs. Level 5).
* **Reliability:** No "Extreme Errors" (e.g., confusing dry Level 0 with saturated Level 10) were observed, making this viable for real-world automated irrigation.


## Technical Specification 
| Parameter | Specification |
| :--- | :--- |
| **Model Architecture** | Vision Transformer (ViT-Base) |
| **Hardware** | Dual NVIDIA T4 GPUs |
| **Optimizer** | AdamW ($5 \times 10^{-5}$ LR) |


## 🏁 Conclusion

This project successfully demonstrates that a **Vision Transformer (ViT)** architecture is highly effective at interpreting the complex spectral patterns created by laser-soil interaction. By achieving a final **Validation Accuracy of 98.11%**, the model proves it can reliably distinguish between 11 different moisture levels (0–10). 

The integration of multi-spectral data (IR, UV, and RGB) allows for a robust classification system that could significantly improve automated irrigation efficiency and water conservation in precision agriculture.

