from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
OUT_DIR = Path("results/confusion_matrices")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# rows = true class, columns = predicted class (from evaluate_models.py output)
MATRICES = {
    "MobileNetV2": [[330, 49, 16, 5], [5, 380, 2, 13], [0, 0, 399, 1], [0, 2, 0, 398]],
    "ResNet50": [[316, 53, 27, 4], [0, 391, 5, 4], [0, 0, 400, 0], [0, 2, 0, 398]],
    "DenseNet121": [[307, 66, 20, 7], [9, 357, 3, 31], [0, 1, 399, 0], [0, 3, 0, 397]],
    "EfficientNetB0": [[330, 44, 18, 8], [6, 381, 0, 13], [0, 1, 399, 0], [0, 0, 0, 400]],
}


def draw(ax, cm, title):
    cm = np.array(cm)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    acc = np.trace(cm) / cm.sum() * 100
    ax.set_title(f"{title} (test accuracy {acc:.2f}%)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=9,
                    color="white" if pct[i, j] > 50 else "black")


# one figure per model
for name, cm in MATRICES.items():
    fig, ax = plt.subplots(figsize=(6, 5))
    draw(ax, cm, name)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_confusion_matrix.png", dpi=200)
    plt.close(fig)

# all four in one figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, cm) in zip(axes.ravel(), MATRICES.items()):
    draw(ax, cm, name)
fig.tight_layout()
fig.savefig(OUT_DIR / "all_models_confusion_matrices.png", dpi=200)
plt.close(fig)

print(f"Saved in {OUT_DIR}")