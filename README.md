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

