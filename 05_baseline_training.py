# STAGE 5: Vision Transformer Fine-Tuning
# Original ViT Baseline Training — 10 epochs (overfit)

import evaluate
import numpy as np
from transformers import (
    ViTForImageClassification,
    TrainingArguments,
    DefaultDataCollator,
    Trainer
)

# Model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=11,
    id2label={i: f"Level {i}" for i in range(11)},
    label2id={f"Level {i}": i for i in range(11)},
    ignore_mismatched_sizes=True
)

# Metric
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# Training Arguments — Original Baseline
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=5,
    num_train_epochs=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    remove_unused_columns=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_ds['train'],
    eval_dataset=prepared_ds['validation'],
    processing_class=processor,
    compute_metrics=compute_metrics
)

trainer.train()

# Save model
trainer.save_model('./results/final_model')
processor.save_pretrained('./results/final_model')
print("Baseline model saved!")

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Exact values from baseline training output
train_losses = [1.756171, 1.077581, 0.757115, 0.496666,
                0.378825, 0.330151, 0.301320, 0.278144,
                0.261974, 0.273489]

val_losses = [1.570183, 1.042456, 0.747440, 0.591142,
              0.498698, 0.443029, 0.411059, 0.391569,
              0.379718, 0.376136]

val_accuracies = [0.871921, 0.931034, 0.940887, 0.955665,
                  0.965517, 0.965517, 0.970443, 0.970443,
                  0.970443, 0.970443]

epochs = range(1, 11)
class_names = [f"Level {i}" for i in range(11)]

# Classification Report
print("\n=== BASELINE CLASSIFICATION REPORT ===")
predictions = trainer.predict(prepared_ds['test'])
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids
print(classification_report(y_true, y_pred, target_names=class_names))

# Accuracy Graph
plt.figure(figsize=(10, 5))
plt.plot(epochs, val_accuracies, label='Validation Accuracy',
         marker='o', color='blue')
plt.axhline(y=0.98, color='r', linestyle='--', label='Target (98%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Baseline — Validation Accuracy vs Target')
plt.ylim([0.8, 1.0])
plt.legend()
plt.grid(True)
plt.savefig('accuracy_graph_baseline.png', dpi=150, bbox_inches='tight')
plt.show()
print("Accuracy graph saved!")

# Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label='Training Loss',
         marker='o', color='blue')
plt.plot(epochs, val_losses, label='Validation Loss',
         marker='s', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Baseline — Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve_baseline.png', dpi=150, bbox_inches='tight')
plt.show()
print("Loss curve saved!")

# Confusion Matrix
class_names_full = [f"Soil Moisture Level {i}" for i in range(11)]
plt.figure(figsize=(14, 12))
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=class_names_full)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Baseline Confusion Matrix — Soil Moisture ViT Classifier')
plt.tight_layout()
plt.savefig('confusion_matrix_baseline.png', dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix saved!")
