# 13_laser_pattern_investigation.py
# Laser Pattern Visual Investigation
# Following instructor feedback on soil_moisture_stir_september performance drop
# This script compares laser reflection patterns across all 7 datasets to identify
# root cause of classification limitations

import os
import random
import matplotlib.pyplot as plt
from PIL import Image

SOURCE_DIR = '/kaggle/working/source_data'

datasets_to_compare = [
    'soil_moisture_stir_september',
    'soil_moisture_september',
    'soil-moisture-v4-uv',
    'soil-moisture-ir',
]

fig, axes = plt.subplots(len(datasets_to_compare), 3, figsize=(15, 20))

for row, dataset in enumerate(datasets_to_compare):
    for split in ['test', 'valid', 'train']:
        img_dir = os.path.join(SOURCE_DIR, dataset, split, 'images')
        if os.path.exists(img_dir):
            break

    imgs = [f for f in os.listdir(img_dir)
            if f.endswith(('.jpg', '.jpeg', '.png'))]
    selected = random.sample(imgs, min(3, len(imgs)))

    for col, img_file in enumerate(selected):
        img = Image.open(os.path.join(img_dir, img_file)).convert("RGB")
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{dataset}\n{img_file[:30]}", fontsize=8)
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('/kaggle/working/laser_pattern_comparison.png', dpi=150)
plt.show()
print("Saved: laser_pattern_comparison.png")
