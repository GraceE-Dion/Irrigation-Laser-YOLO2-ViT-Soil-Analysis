#STAGE 7
#the training arguments was changed to address the overfitting of the model by 
#retraining the existing whole-image ViT with the regularization fixes, 
#so you have a clean baseline accuracy improvement to compare against

# Step 8 REVISED — Fixed Augmentation

from PIL import Image as PILImage
from torchvision import transforms
import torch

# Augmentation ONLY — no ToTensor or Normalize here
train_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomResizedCrop(
        224,
        scale=(0.7, 1.0),
        ratio=(0.8, 1.2)
    ),
    transforms.GaussianBlur(
        kernel_size=3,
        sigma=(0.1, 2.0)
    ),
    transforms.RandomAdjustSharpness(
        sharpness_factor=2,
        p=0.3
    ),
    # NO ToTensor or Normalize — processor handles this
])

def transform_train(example_batch):
    augmented_images = [
        train_augmentation(img.convert("RGB")) 
        for img in example_batch['image']
    ]
    inputs = processor(
        images=augmented_images,
        return_tensors='pt'
    )
    inputs['labels'] = example_batch['label']
    return inputs

def transform_val(example_batch):
    inputs = processor(
        images=[img.convert("RGB") for img in example_batch['image']],
        return_tensors='pt'
    )
    inputs['labels'] = example_batch['label']
    return inputs

# Apply transforms
prepared_ds_train = raw_ds['train'].with_transform(transform_train)
prepared_ds_val   = raw_ds['validation'].with_transform(transform_val)
prepared_ds_test  = raw_ds['test'].with_transform(transform_val)

print("Augmentation pipeline ready!")

# Step 9 REVISED — Phase 2: Training with Augmentation
import evaluate
import numpy as np
from transformers import (
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# Model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=11,
    id2label={i: f"Level {i}" for i in range(11)},
    label2id={f"Level {i}": i for i in range(11)},
    ignore_mismatched_sizes=True,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)

# Metric
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# Training Arguments — Phase 2 UPDATED
training_args = TrainingArguments(
    output_dir="./results_v3",
    save_total_limit=1,
    save_strategy="no",              # no checkpoints during training
    load_best_model_at_end=False,    # must be False when save_strategy="no"
    eval_strategy="epoch",
    logging_steps=5,
    num_train_epochs=25,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    metric_for_best_model="accuracy",
    greater_is_better=True,
    remove_unused_columns=False,
    label_smoothing_factor=0.1,
)

# Trainer — no EarlyStoppingCallback since save_strategy="no"
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_ds_train,
    eval_dataset=prepared_ds_val,
    processing_class=processor,
    compute_metrics=compute_metrics,
)
trainer.train()
# Save once after training completes
trainer.save_model('./results_v3/final_model')
processor.save_pretrained('./results_v3/final_model')
print("Model saved!")

