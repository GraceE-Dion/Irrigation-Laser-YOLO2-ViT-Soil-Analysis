import torch
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# 1. Setup Device & Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./models/vit_soil_moisture_final" 
model = ViTForImageClassification.from_pretrained(model_path).to(device)
processor = ViTImageProcessor.from_pretrained(model_path)
model.eval()

# --- PART A: GENERATE TRAINING CURVES ---
def save_performance_curves(history_csv):
    df = pd.read_csv(history_csv) # Assumes your Kaggle log is saved as CSV
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['accuracy'], label='Train Acc', color='#1f77b4', lw=2)
    plt.plot(df['epoch'], df['val_accuracy'], label='Val Acc', color='#ff7f0e', lw=2)
    plt.axhline(y=0.9811, color='gold', linestyle='--', label='Target (98.11%)')
    plt.title('Model Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['loss'], label='Train Loss', color='#d62728', lw=2)
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#2ca02c', lw=2)
    plt.title('Model Loss (Cross-Entropy)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('images/training_metrics.png', dpi=300)
    print("✅ Training curves saved to images/training_metrics.png")

# --- PART B: 7-SOURCE INFERENCE TEST ---
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# 1. Setup & Load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./models/vit_soil_moisture_final"
model = ViTForImageClassification.from_pretrained(model_path).to(device)
processor = ViTImageProcessor.from_pretrained(model_path)
model.eval()

# 2. Select 3 Random Images from your Test Set
# (Adjust 'test_folder' to your actual Kaggle input path)
import os
test_folder = "/kaggle/input/soil-moisture-v4/test/images"
all_images = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith(('.jpg', '.png'))]
selected_images = random.sample(all_images, 3)

# 3. Plotting Setup
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Real-World Inference: ViT Multi-Spectral Validation', fontsize=20, y=1.05)

for i, img_path in enumerate(selected_images):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        conf, pred = torch.max(probs, dim=1)

    # Visualization
    axes[i].imshow(image)
    axes[i].set_title(f"Pred: Level {pred.item()}\nConf: {conf.item()*100:.2f}%", 
                      fontsize=14, color='green' if conf.item() > 0.9 else 'orange')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('inference_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Inference visualization saved as inference_results.png")
