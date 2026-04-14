#STAGE 9: Physical Noise Augmentation on Laser Crops
#
# Background:
# Phase 3 achieved 87.68% accuracy using cropped laser regions but
# mid-range moisture levels (3, 4, 6) showed persistent confusion.
# Following instructor guidance, Phase 4A implements physical noise
# augmentation — generating and saving Gaussian noise and salt-and-pepper
# noise copies of each training image to disk, effectively tripling the
# training set from 717 to 2,151 images.
#
# Unlike on-the-fly augmentation used in Phase 2, this approach physically
# creates new image files that are saved alongside the originals before
# reloading the dataset. This ensures the model trains on the combined
# original and noisy data together, exposing it to a wider variety of
# laser diffusion patterns across different noise conditions.
#
# Key improvement over Phase 3:
# - Training set tripled from 717 to 2,151 images
# - Level 10 F1 score improved from 0.80 to 1.00
# - Level 2 F1 score improved from 0.64 to 0.80
# - Overall accuracy improved from 87.68% to 89.66%

# Step 16 — Phase 4A: Generate Noisy Augmented Images
import os
import numpy as np
from PIL import Image
import random

LASER_DIR = '/kaggle/working/Master_Laser_Crops'
train_path = os.path.join(LASER_DIR, 'train')

def add_gaussian_noise(img, mean=0, std=25):
    """Add Gaussian noise to image."""
    img_array = np.array(img).astype(np.float32)
    noise = np.random.normal(mean, std, img_array.shape)
    noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def add_salt_pepper_noise(img, amount=0.05):
    """Add salt and pepper noise to image."""
    img_array = np.array(img).astype(np.uint8)
    noisy = img_array.copy()
    
    # Salt — white pixels
    num_salt = int(amount * img_array.size * 0.5)
    salt_coords = [np.random.randint(0, i, num_salt) 
                   for i in img_array.shape[:2]]
    noisy[salt_coords[0], salt_coords[1]] = 255
    
    # Pepper — black pixels
    num_pepper = int(amount * img_array.size * 0.5)
    pepper_coords = [np.random.randint(0, i, num_pepper) 
                     for i in img_array.shape[:2]]
    noisy[pepper_coords[0], pepper_coords[1]] = 0
    
    return Image.fromarray(noisy)

def flip_image(img):
    """Apply random horizontal or vertical flip."""
    choice = random.randint(0, 2)
    if choice == 0:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif choice == 1:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        # Both flips
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img.transpose(Image.FLIP_TOP_BOTTOM)

# Generate augmented copies
augmented_count = 0

for class_folder in os.listdir(train_path):
    class_path = os.path.join(train_path, class_folder)
    if not os.path.isdir(class_path):
        continue

    original_files = [f for f in os.listdir(class_path)
                      if f.endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in original_files:
        img_path = os.path.join(class_path, img_file)
        img = Image.open(img_path).convert("RGB")
        base_name = img_file.rsplit('.', 1)[0]

        # Copy 1 — flip + Gaussian noise
        aug1 = flip_image(img)
        aug1 = add_gaussian_noise(aug1, mean=0, std=25)
        aug1_path = os.path.join(class_path, f"{base_name}_aug_gaussian.jpg")
        aug1.save(aug1_path)

        # Copy 2 — flip + Salt & Pepper noise
        aug2 = flip_image(img)
        aug2 = add_salt_pepper_noise(aug2, amount=0.05)
        aug2_path = os.path.join(class_path, f"{base_name}_aug_saltpepper.jpg")
        aug2.save(aug2_path)

        augmented_count += 2

print(f"Augmentation complete!")
print(f"Original images: 717")
print(f"New augmented images added: {augmented_count}")
print(f"Total training images: {717 + augmented_count}")

# Verify counts per class
print("\nPer class breakdown:")
for class_folder in sorted(os.listdir(train_path)):
    class_path = os.path.join(train_path, class_folder)
    if os.path.isdir(class_path):
        count = len(os.listdir(class_path))
        print(f"  Class {class_folder}: {count} images")

Step 17: Reload Expanded Dataset

from datasets import load_dataset
from datasets import Image as HFImage

# Reload laser dataset — now includes augmented images
laser_ds_aug = load_dataset(
    "imagefolder",
    data_dir=LASER_DIR,
    drop_labels=False
)
laser_ds_aug = laser_ds_aug.cast_column("image", HFImage(decode=True))
print(laser_ds_aug)
print(f"Training set expanded to: {laser_ds_aug['train'].num_rows} images")

Step 18: Apply Transforms to Expanded Dataset

from datasets import load_dataset
from datasets import Image as HFImage

# Reload laser dataset — now includes augmented images
laser_ds_aug = load_dataset(
    "imagefolder",
    data_dir=LASER_DIR,
    drop_labels=False
)
laser_ds_aug = laser_ds_aug.cast_column("image", HFImage(decode=True))
print(laser_ds_aug)
print(f"Training set expanded to: {laser_ds_aug['train'].num_rows} images")

Step 19: Train ViT on Augmented Laser Crops

import evaluate
import numpy as np
from transformers import (
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)

# Fresh model
model_v4 = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=11,
    id2label={i: f"Level {i}" for i in range(11)},
    label2id={f"Level {i}": i for i in range(11)},
    ignore_mismatched_sizes=True,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

training_args_v4 = TrainingArguments(
    output_dir="./results_v5",
    save_strategy="no",
    eval_strategy="epoch",
    logging_steps=5,
    num_train_epochs=40,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    load_best_model_at_end=False,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    remove_unused_columns=False,
    label_smoothing_factor=0.1,
    save_total_limit=1,
)

trainer_v4 = Trainer(
    model=model_v4,
    args=training_args_v4,
    train_dataset=laser_train_aug,
    eval_dataset=laser_val_aug,
    processing_class=processor,
    compute_metrics=compute_metrics,
)

trainer_v4.train()
# Save final model
trainer_v4.save_model('./results_v5/final_model')
processor.save_pretrained('./results_v5/final_model')
print("Phase 4A model saved!")
