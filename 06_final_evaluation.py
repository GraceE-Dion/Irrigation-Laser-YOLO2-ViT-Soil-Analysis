# STAGE 6: Validation & Results
# Generates the Confusion Matrix and Classification Report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Save the results to a PNG for GitHub
plt.savefig('final_results_matrix.png')
