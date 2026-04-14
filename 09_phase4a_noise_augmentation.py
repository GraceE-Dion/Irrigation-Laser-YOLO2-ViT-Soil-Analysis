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

# Step 20A: Phase 4A Metrics
# Fix — import missing and replot confusion matrix only
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class_names_full = [f"Soil Moisture Level {i}" for i in range(11)]
plt.figure(figsize=(14, 12))
cm_p4a = confusion_matrix(y_true_p4a, y_pred_p4a)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_p4a,
                               display_labels=class_names_full)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Phase 4A Confusion Matrix — Augmented Laser Crop ViT Classifier')
plt.tight_layout()
plt.savefig('confusion_matrix_phase4a.png', dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix saved!")
train_losses_p4a = [4.598878, 4.018516, 3.466031, 3.105857, 2.701540,
                    2.416365, 2.183443, 1.930872, 1.865558, 1.660936,
                    1.601121, 1.554153, 1.487123, 1.435534, 1.452652,
                    1.371130, 1.396241, 1.277487, 1.367294, 1.145833,
                    1.172555, 1.357800, 1.288642, 1.175231, 1.273762,
                    1.203880, 1.224792, 1.190668, 1.156583, 1.123374,
                    1.139964, 1.203056, 1.173595, 1.167897, 1.171117,
                    1.172913, 1.196974, 1.106842, 1.134756, 1.133870]

val_losses_p4a = [4.553881, 3.976180, 3.464017, 2.997657, 2.627634,
                  2.384494, 2.262010, 2.055072, 1.935329, 1.903518,
                  1.802516, 1.753705, 1.762875, 1.825698, 1.669055,
                  1.766719, 1.687311, 1.614212, 1.633559, 1.572980,
                  1.654490, 1.615139, 1.625992, 1.633548, 1.589483,
                  1.614638, 1.604377, 1.591931, 1.569810, 1.570749,
                  1.570167, 1.573008, 1.596105, 1.591208, 1.593277,
                  1.589233, 1.593882, 1.597016, 1.597456, 1.597225]

val_accuracies_p4a = [0.216749, 0.344828, 0.566502, 0.778325, 0.812808,
                      0.812808, 0.778325, 0.842365, 0.842365, 0.822660,
                      0.842365, 0.852217, 0.857143, 0.837438, 0.876847,
                      0.847291, 0.871921, 0.886700, 0.876847, 0.896552,
                      0.871921, 0.876847, 0.891626, 0.876847, 0.886700,
                      0.876847, 0.881773, 0.891626, 0.896552, 0.891626,
                      0.881773, 0.891626, 0.886700, 0.881773, 0.881773,
                      0.886700, 0.886700, 0.881773, 0.886700, 0.886700]

epochs_p4a = range(1, 41)
class_names = [f"Level {i}" for i in range(11)]

# Classification Report
print("\n=== PHASE 4A CLASSIFICATION REPORT ===")
predictions_p4a = trainer_v4.predict(laser_test_aug)
y_pred_p4a = np.argmax(predictions_p4a.predictions, axis=1)
y_true_p4a = predictions_p4a.label_ids
print(classification_report(y_true_p4a, y_pred_p4a, target_names=class_names))

# Accuracy Graph
plt.figure(figsize=(10, 5))
plt.plot(epochs_p4a, val_accuracies_p4a, label='Validation Accuracy',
         marker='o', color='blue')
plt.axhline(y=0.98, color='r', linestyle='--', label='Target (98%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Phase 4A — Validation Accuracy vs Target')
plt.ylim([0.1, 1.0])
plt.legend()
plt.grid(True)
plt.savefig('accuracy_graph_phase4a.png', dpi=150, bbox_inches='tight')
plt.show()

# Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(epochs_p4a, train_losses_p4a, label='Training Loss',
         marker='o', color='blue')
plt.plot(epochs_p4a, val_losses_p4a, label='Validation Loss',
         marker='s', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Phase 4A — Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve_phase4a.png', dpi=150, bbox_inches='tight')
plt.show()

# Confusion Matrix
class_names_full = [f"Soil Moisture Level {i}" for i in range(11)]
plt.figure(figsize=(14, 12))
cm_p4a = confusion_matrix(y_true_p4a, y_pred_p4a)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_p4a,
                               display_labels=class_names_full)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Phase 4A Confusion Matrix — Augmented Laser Crop ViT Classifier')
plt.tight_layout()
plt.savefig('confusion_matrix_phase4a.png', dpi=150, bbox_inches='tight')
plt.show()
print("Phase 4A metrics saved!")
