# STAGE 6: Validation & Results
# Generates the Confusion Matrix, Accuracy graph, loss curve and Classification Report
# Phase 1: Overfitting Fix — 17 epochs, dropout, early stopping

import evaluate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import (
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# Model with dropout regularization
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

# Training Arguments — Phase 1
training_args = TrainingArguments(
    output_dir="./results_v2",
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=5,
    num_train_epochs=20,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    remove_unused_columns=False,
    save_total_limit=1,
)

# Trainer with Early Stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_ds['train'],
    eval_dataset=prepared_ds['validation'],
    processing_class=processor,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )]
)

trainer.train()

# Save model
trainer.save_model('./results_v2/final_model')
processor.save_pretrained('./results_v2/final_model')
print("Phase 1 model saved!")

# Phase 1 actual training values — 17 epochs
train_losses_p1 = [2.344809, 2.118603, 1.936597, 1.603722, 1.359488,
                   1.207903, 1.005909, 0.903221, 0.784347, 0.738066,
                   0.660259, 0.600291, 0.564556, 0.526923, 0.508441,
                   0.523409, 0.499586]

val_losses_p1 = [2.318297, 2.081824, 1.797832, 1.574416, 1.370550,
                 1.195082, 1.046510, 0.922475, 0.825795, 0.748198,
                 0.666923, 0.640123, 0.605296, 0.580547, 0.567832,
                 0.554361, 0.547682]

val_accuracies_p1 = [0.270936, 0.448276, 0.674877, 0.862069, 0.871921,
                     0.886700, 0.940887, 0.950739, 0.950739, 0.955665,
                     0.960591, 0.960591, 0.960591, 0.965517, 0.965517,
                     0.965517, 0.965517]

epochs_p1 = range(1, 18)
class_names = [f"Level {i}" for i in range(11)]

# Classification Report
print("\n=== PHASE 1 CLASSIFICATION REPORT ===")
predictions = trainer.predict(prepared_ds['test'])
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids
print(classification_report(y_true, y_pred, target_names=class_names))

# Accuracy Graph
plt.figure(figsize=(10, 5))
plt.plot(epochs_p1, val_accuracies_p1, label='Validation Accuracy',
         marker='o', color='blue')
plt.axhline(y=0.98, color='r', linestyle='--', label='Target (98%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Phase 1 — Validation Accuracy vs Target')
plt.ylim([0.2, 1.0])
plt.legend()
plt.grid(True)
plt.savefig('accuracy_graph_phase1.png', dpi=150, bbox_inches='tight')
plt.show()
print("Phase 1 accuracy graph saved!")

# Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(epochs_p1, train_losses_p1, label='Training Loss',
         marker='o', color='blue')
plt.plot(epochs_p1, val_losses_p1, label='Validation Loss',
         marker='s', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Phase 1 — Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve_phase1.png', dpi=150, bbox_inches='tight')
plt.show()
print("Phase 1 loss curve saved!")

# Confusion Matrix
class_names_full = [f"Soil Moisture Level {i}" for i in range(11)]
plt.figure(figsize=(14, 12))
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=class_names_full)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Phase 1 Confusion Matrix — Soil Moisture ViT Classifier')
plt.tight_layout()
plt.savefig('confusion_matrix_phase1.png', dpi=150, bbox_inches='tight')
plt.show()
print("Phase 1 confusion matrix saved!")
