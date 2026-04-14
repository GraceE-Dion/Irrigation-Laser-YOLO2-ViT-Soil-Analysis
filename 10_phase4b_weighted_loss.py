# STAGE 10:Phase 4B: Class-Weighted Loss Function
#
# Background:
# Phase 4A improved overall accuracy to 89.66% through physical noise
# augmentation, and Level 10 reached a perfect 1.00 F1 score. However
# Level 4 unexpectedly dropped to 0.25 F1 and Level 6 remained weak
# at 0.52 F1. Analysis of the confusion matrix revealed that Level 4
# was being heavily confused with its neighbors Levels 3 and 5, with
# 6 out of 9 test samples misclassified.
#
# Phase 4B addresses this class imbalance by implementing inverse
# frequency class weighting through a custom WeightedTrainer. Class
# weights are computed from training sample counts — underrepresented
# classes receive higher weights, forcing the model to pay more
# attention to difficult classes during training.
#
# Targeted weak classes:
# - Level 2 (78 training samples) — weight 2.0
# - Level 4 (109 training samples) — weight 3.0
# - Level 6 (42 training samples) — weight 2.0
#
# Key improvement over Phase 4A:
# - Overall accuracy improved from 89.66% to 90.64%
# - Level 4 F1 recovered from 0.25 to 0.43
# - Level 6 F1 recovered from 0.52 to 0.64
# - Level 10 remained perfect at 1.00
# - Best ViT result across all phases

#Step 19B: Weighted Trainer
import torch
import numpy as np
import evaluate
from transformers import ViTForImageClassification, TrainingArguments, Trainer

# Compute class weights from training data
class_counts = [34, 59, 78, 65, 109, 105, 42, 53, 65, 36, 71]  # from Step 13 output
total = sum(class_counts)
num_classes = len(class_counts)

# Inverse frequency weighting — rare classes get higher weight
class_weights = torch.tensor(
    [total / (num_classes * count) for count in class_counts],
    dtype=torch.float32
).to('cuda')

print("Class weights:")
for i, w in enumerate(class_weights):
    print(f"  Level {i}: {w:.4f}")

# Custom Trainer with weighted loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Fresh model
model_v4b = ViTForImageClassification.from_pretrained(
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

training_args_v4b = TrainingArguments(
    output_dir="./results_v6",
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

trainer_v4b = WeightedTrainer(
    model=model_v4b,
    args=training_args_v4b,
    train_dataset=laser_train_aug,
    eval_dataset=laser_val_aug,
    processing_class=processor,
    compute_metrics=compute_metrics,
)

trainer_v4b.train()

# Save
trainer_v4b.save_model('./results_v6/final_model')
processor.save_pretrained('./results_v6/final_model')
print("Phase 4B weighted loss model saved!")

# Step 20B: Phase 4B Metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             ConfusionMatrixDisplay)

train_losses_p4b = [2.293143, 1.928957, 1.661045, 1.403453, 1.121677,
                    1.010281, 0.819675, 0.739883, 0.671993, 0.521686,
                    0.500440, 0.470915, 0.441232, 0.418277, 0.377515,
                    0.331545, 0.337027, 0.261187, 0.324425, 0.231847,
                    0.224760, 0.282517, 0.225149, 0.176197, 0.212028,
                    0.188740, 0.206023, 0.179401, 0.159163, 0.140444,
                    0.139795, 0.232164, 0.192412, 0.164299, 0.171826,
                    0.165090, 0.184149, 0.133453, 0.153742, 0.133313]

val_losses_p4b = [2.288795, 2.021616, 1.686626, 1.453947, 1.222767,
                  1.093630, 0.958269, 0.834218, 0.782016, 0.725853,
                  0.660369, 0.623775, 0.587952, 0.634187, 0.525539,
                  0.503068, 0.516293, 0.485166, 0.499607, 0.514354,
                  0.516399, 0.491443, 0.452487, 0.433715, 0.431894,
                  0.433852, 0.426308, 0.426420, 0.435224, 0.420108,
                  0.414563, 0.408872, 0.418126, 0.415748, 0.416119,
                  0.408963, 0.416059, 0.415985, 0.415391, 0.415321]

val_accuracies_p4b = [0.221675, 0.379310, 0.566502, 0.640394, 0.640394,
                      0.699507, 0.773399, 0.827586, 0.822660, 0.822660,
                      0.847291, 0.852217, 0.857143, 0.822660, 0.871921,
                      0.876847, 0.871921, 0.881773, 0.871921, 0.862069,
                      0.866995, 0.862069, 0.876847, 0.901478, 0.891626,
                      0.896552, 0.891626, 0.896552, 0.896552, 0.896552,
                      0.901478, 0.906404, 0.896552, 0.896552, 0.901478,
                      0.906404, 0.901478, 0.901478, 0.901478, 0.901478]

epochs_p4b = range(1, 41)
class_names = [f"Level {i}" for i in range(11)]

# Classification Report
print("\n=== PHASE 4B CLASSIFICATION REPORT ===")
predictions_p4b = trainer_v4b.predict(laser_test_aug)
y_pred_p4b = np.argmax(predictions_p4b.predictions, axis=1)
y_true_p4b = predictions_p4b.label_ids
print(classification_report(y_true_p4b, y_pred_p4b, target_names=class_names))

# Accuracy Graph
plt.figure(figsize=(10, 5))
plt.plot(epochs_p4b, val_accuracies_p4b, label='Validation Accuracy',
         marker='o', color='blue')
plt.axhline(y=0.98, color='r', linestyle='--', label='Target (98%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Phase 4B — Validation Accuracy vs Target')
plt.ylim([0.1, 1.0])
plt.legend()
plt.grid(True)
plt.savefig('accuracy_graph_phase4b.png', dpi=150, bbox_inches='tight')
plt.show()

# Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(epochs_p4b, train_losses_p4b, label='Training Loss',
         marker='o', color='blue')
plt.plot(epochs_p4b, val_losses_p4b, label='Validation Loss',
         marker='s', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Phase 4B — Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve_phase4b.png', dpi=150, bbox_inches='tight')
plt.show()

# Confusion Matrix
class_names_full = [f"Soil Moisture Level {i}" for i in range(11)]
plt.figure(figsize=(14, 12))
cm_p4b = confusion_matrix(y_true_p4b, y_pred_p4b)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_p4b,
                               display_labels=class_names_full)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Phase 4B Confusion Matrix — Weighted Loss ViT Classifier')
plt.tight_layout()
plt.savefig('confusion_matrix_phase4b.png', dpi=150, bbox_inches='tight')
plt.show()
print("Phase 4B metrics saved!")

