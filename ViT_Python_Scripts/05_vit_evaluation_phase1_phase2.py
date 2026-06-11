"""
Script 05: Evaluation — Phase 1 and Phase 2
Loss curves, accuracy graphs, classification report, confusion matrix
Soil Moisture Classification — ViT + YOLOv8 Pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ── Hardcoded training history (Phase 2 actual values) ────────────────────
train_losses = [4.673071, 4.453415, 4.095893, 3.647770, 3.428395,
                3.116595, 2.688093, 2.507006, 2.278150, 2.079329,
                1.984986, 1.814550, 1.846718, 1.590630, 1.656840,
                1.560463, 1.605761, 1.555979, 1.414505, 1.446769,
                1.404795, 1.368590, 1.344135, 1.435816, 1.418155]

val_losses = [4.751688, 4.460595, 4.137558, 3.761130, 3.428313,
              3.167595, 2.809613, 2.599695, 2.414394, 2.290302,
              2.129147, 1.914390, 1.913347, 1.790328, 1.751759,
              1.676690, 1.670827, 1.611479, 1.599333, 1.581134,
              1.574428, 1.565577, 1.563511, 1.564832, 1.564560]

val_accuracies = [0.133005, 0.339901, 0.389163, 0.453202, 0.556650,
                  0.748768, 0.837438, 0.857143, 0.876847, 0.862069,
                  0.866995, 0.940887, 0.886700, 0.896552, 0.921182,
                  0.945813, 0.926108, 0.945813, 0.945813, 0.945813,
                  0.945813, 0.945813, 0.945813, 0.945813, 0.945813]

epochs = range(1, 26)
class_names = [f"Level_{i}" for i in range(11)]

# ── Classification report ─────────────────────────────────────────────────
print("=== CLASSIFICATION REPORT ===")
predictions = trainer.predict(prepared_ds_test)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids
print(classification_report(y_true, y_pred, target_names=class_names))

# ── Accuracy graph ─────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(epochs, val_accuracies, label='Validation Accuracy',
         marker='o', color='blue')
plt.axhline(y=0.98, color='r', linestyle='--', label='Target (98%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy vs Target — Phase 2')
plt.ylim([0.8, 1.0])
plt.legend()
plt.grid(True)
plt.savefig('accuracy_graph_phase2.png', dpi=150, bbox_inches='tight')
plt.show()
print("Accuracy graph saved.")

# ── Loss curve ────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label='Training Loss', marker='o', color='blue')
plt.plot(epochs, val_losses,   label='Validation Loss', marker='s', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve — Phase 2')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve_phase2.png', dpi=150, bbox_inches='tight')
plt.show()
print("Loss curve saved.")

# ── Confusion matrix ──────────────────────────────────────────────────────
class_names_full = [f"Soil Moisture Level {i}" for i in range(11)]

plt.figure(figsize=(14, 12))
cm   = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=class_names_full)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix — Soil Moisture ViT Classifier')
plt.tight_layout()
plt.savefig('confusion_matrix_phase2.png', dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix saved.")
