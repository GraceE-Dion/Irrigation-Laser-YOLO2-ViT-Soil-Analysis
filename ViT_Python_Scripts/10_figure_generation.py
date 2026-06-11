"""
Script 10: Publication-quality figure generation
Figures 1, 2, 3, 4, 5, 6, 7 for journal submission
Soil Moisture Classification — ViT + YOLOv8 Pipeline
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import glob
import shutil
import pandas as pd
from PIL import Image
from IPython.display import FileLink, display

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 14,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'font.family': 'DejaVu Sans',
})

SOURCE_DIR    = '/kaggle/working/source_data'
YOLO_RESULTS  = '/kaggle/working/yolo_results/soil_moisture_yolo'
INFERENCE_DIR = '/kaggle/working/inference_phase6B'
OUT           = '/kaggle/output/paper_figures_large'
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/{name}.{ext}', dpi=300,
                    bbox_inches='tight', facecolor='white')
    shutil.copy(f'{OUT}/{name}.png', f'/kaggle/working/{name}.png')
    shutil.copy(f'{OUT}/{name}.pdf', f'/kaggle/working/{name}.pdf')
    print(f'Saved {name}')
    plt.close(fig)

# ── Figure 4: Laser Pattern Visual Comparison ─────────────────────────────
print("Generating Figure 4...")

panel_info = [
    ('soil_moisture_stir_september', 'IR Laser — Uncontrolled Field',
     'Dim/invisible laser spot\nPhase 6 accuracy: 20%', 'red'),
    ('soil_moisture_september',      'UV Laser — Field',
     'Consistent blue UV glow\nPhase 6 accuracy: 57%', 'orange'),
    ('soil-moisture-v4-uv',          'UV Laser — Controlled Lab',
     'Strong, bright UV spot\nPhase 6 accuracy: 100%', '#2E7D5E'),
    ('soil-moisture-ir',             'IR Laser — Controlled Lab',
     'Clear white IR spot\nPhase 6 accuracy: 100%', '#2E7D5E'),
]

fig, axes = plt.subplots(1, 4, figsize=(24, 8))
fig.suptitle(
    'Laser-Pattern Visual Comparison: '
    'Capture Environment Governs Performance, Not Wavelength',
    fontsize=20, fontweight='bold', y=1.02
)

for ax, (ds, modality, caption, border_col) in zip(axes, panel_info):
    img_path = None
    for split in ['test', 'valid', 'train']:
        for ext in ['*.jpg', '*.png']:
            matches = glob.glob(
                os.path.join(SOURCE_DIR, ds, split, 'images', ext))
            if matches:
                img_path = matches[0]
                break
        if img_path:
            break

    if img_path:
        ax.imshow(Image.open(img_path))
    else:
        ax.set_facecolor('#222222')
        ax.text(0.5, 0.5, f'[{ds}]\nimage not found',
                ha='center', va='center', color='white',
                fontsize=12, transform=ax.transAxes)

    ax.set_title(f'{ds}\n{modality}', fontsize=14,
                 fontweight='bold', color=border_col, pad=8)
    ax.set_xlabel(caption, fontsize=13, color=border_col, labelpad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(border_col)
        spine.set_linewidth(4)

fig.text(0.5, -0.04,
         'Left to right: IR uncontrolled field (20%), UV field (57%), '
         'UV controlled lab (100%), IR controlled lab (100%)',
         ha='center', fontsize=14, style='italic')
fig.tight_layout()
save(fig, 'fig4_laser_pattern_comparison')

# ── Figure 5: Phase 6 YOLOv8 Training Curves ─────────────────────────────
print("Generating Figure 5...")

results_path = os.path.join(YOLO_RESULTS, 'results.csv')
if not os.path.exists(results_path):
    print(f"WARNING: results.csv not found at {results_path} — skipping Fig 5")
else:
    df = pd.read_csv(results_path)
    df.columns = df.columns.str.strip()
    best_ep = 32

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        'Phase 6 YOLOv8 Training Curves — '
        'Best Epoch 32/42 (EarlyStopping patience=10)',
        fontsize=20, fontweight='bold'
    )

    def plot_panel(ax, col_train, col_val, title, ylabel, best_epoch):
        cols      = df.columns.tolist()
        train_col = next((c for c in cols if col_train.lower() in c.lower()), None)
        val_col   = next((c for c in cols if col_val.lower()   in c.lower()), None)
        if train_col:
            ax.plot(df[train_col], label='Train', color='#4C72B0', linewidth=2.5)
        if val_col:
            ax.plot(df[val_col],   label='Val',   color='#C0392B', linewidth=2.5)
        ax.axvline(best_epoch, color='#888888', linestyle='--',
                   linewidth=2.0, label=f'Best epoch {best_epoch}')
        if title == 'mAP50' and val_col:
            best_val = df[val_col].max()
            ax.annotate(
                f'Best: {best_val:.3f} ({best_val*100:.1f}%)',
                xy=(best_epoch, best_val),
                xytext=(best_epoch + 2, best_val - 0.05),
                fontsize=13, color='#333333',
                arrowprops=dict(arrowstyle='->', lw=1.5)
            )
        ax.set_title(title, fontsize=17, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.legend(fontsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)

    plot_panel(axes[0, 0], 'box_loss',          'val/box',
               'Box Loss',            'Loss',  best_ep)
    plot_panel(axes[0, 1], 'cls_loss',          'val/cls',
               'Classification Loss', 'Loss',  best_ep)
    plot_panel(axes[1, 0], 'metrics/mAP50',     'metrics/mAP50',
               'mAP50',               'mAP50', best_ep)
    plot_panel(axes[1, 1], 'metrics/precision', 'metrics/recall',
               'Precision and Recall', 'Value', best_ep)

    fig.tight_layout()
    save(fig, 'fig5_phase6_training_curves')

# ── Figure 6: Phase 6 Normalised Confusion Matrix ─────────────────────────
print("Generating Figure 6...")

cm_path = os.path.join(YOLO_RESULTS, 'confusion_matrix_normalized.png')
if not os.path.exists(cm_path):
    print(f"WARNING: confusion_matrix_normalized.png not found — skipping Fig 6")
else:
    cm_img = Image.open(cm_path)
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(cm_img)
    ax.axis('off')
    ax.set_title(
        'Phase 6 Normalised Confusion Matrix — YOLOv8 (95.3% mAP50)',
        fontsize=18, fontweight='bold', pad=15
    )
    fig.tight_layout()
    save(fig, 'fig6_confusion_matrix')

# ── Figure 7: Per-dataset inference output grid ───────────────────────────
print("Generating Figure 7 — 7 separate dataset files...")

dataset_order = [
    'soil-moisture-v4',       'soil-moisture-v4-ir',
    'soil-moisture-v4-uv',    'soil-moisture-ir',
    'soil-moisture-5sagf',    'soil_moisture_september',
    'soil_moisture_stir_september',
]
dataset_labels = {
    'soil-moisture-v4':             'Dataset 1: soil-moisture-v4 (Standard RGB)',
    'soil-moisture-v4-ir':          'Dataset 2: soil-moisture-v4-ir (Infrared)',
    'soil-moisture-v4-uv':          'Dataset 3: soil-moisture-v4-uv (Ultraviolet)',
    'soil-moisture-ir':             'Dataset 4: soil-moisture-ir (Infrared Controlled)',
    'soil-moisture-5sagf':          'Dataset 5: soil-moisture-5sagf (General Field)',
    'soil_moisture_september':      'Dataset 6: soil-moisture-september (Seasonal UV)',
    'soil_moisture_stir_september': 'Dataset 7: soil-moisture-stir-september (IR Stirred)',
}
output_names = {
    'soil-moisture-v4':             'inference_soil_moisture_v4',
    'soil-moisture-v4-ir':          'inference_soil_moisture_v4_ir',
    'soil-moisture-v4-uv':          'inference_soil_moisture_v4_uv',
    'soil-moisture-ir':             'inference_soil_moisture_ir',
    'soil-moisture-5sagf':          'inference_soil_moisture_5sagf',
    'soil_moisture_september':      'inference_soil_moisture_september',
    'soil_moisture_stir_september': 'inference_soil_moisture_stir_september',
}

all_files = sorted(
    glob.glob(f'{INFERENCE_DIR}/*.jpg') +
    glob.glob(f'{INFERENCE_DIR}/*.png')
)

grouped = {d: [] for d in dataset_order}
for f in all_files:
    fname = os.path.basename(f)
    for d in sorted(dataset_order, key=len, reverse=True):
        if d in fname:
            grouped[d].append(f)
            break

for d in dataset_order:
    files = grouped[d]
    if not files:
        print(f'No images found for {d} — skipping')
        continue

    # Classify correct/incorrect by green vs red pixel in bottom strip
    correct   = []
    incorrect = []
    for f in files:
        img_arr      = np.array(Image.open(f))
        bottom_strip = img_arr[-60:, :, :]
        green_pixels = np.sum(
            (bottom_strip[:, :, 1] > 150) &
            (bottom_strip[:, :, 0] < 100) &
            (bottom_strip[:, :, 2] < 100)
        )
        red_pixels = np.sum(
            (bottom_strip[:, :, 0] > 150) &
            (bottom_strip[:, :, 1] < 100) &
            (bottom_strip[:, :, 2] < 100)
        )
        if green_pixels > red_pixels:
            correct.append(f)
        else:
            incorrect.append(f)

    # Show 2 correct + 1 incorrect per dataset
    picks = correct[:2] + incorrect[:1]
    remaining = [f for f in files if f not in picks]
    while len(picks) < 3 and remaining:
        picks.append(remaining.pop(0))
    picks = picks[:3]

    ncols = len(picks)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 7, 10))
    if ncols == 1:
        axes = [axes]

    fig.suptitle(dataset_labels[d], fontsize=16, fontweight='bold', y=1.01)

    for ax, img_file in zip(axes, picks):
        ax.imshow(Image.open(img_file))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    save(fig, output_names[d])

print(f"\nAll figures saved to {OUT}")
