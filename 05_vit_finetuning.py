# STAGE 5: Vision Transformer Fine-Tuning
from transformers import ViTForImageClassification, TrainingArguments, Trainer

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=11
)
# Training with 10 Epochs and Learning Rate 5e-5
