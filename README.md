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

 ![Confusion Matrix](https://github.com/user-attachments/assets/69d61f5d-49c4-42de-9fa1-5e2067125687)
