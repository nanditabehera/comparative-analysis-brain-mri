from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
OUT_DIR = Path("results/roc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "MobileNetV2": ("models/mobilenetv2_best_94_19.keras", None),
    "ResNet50": ("models/resnet50_best_94_06.keras",
                 keras.applications.resnet50.preprocess_input),
    "DenseNet121": ("models/densenet121_best_91_25.keras",
                    keras.applications.densenet.preprocess_input),
    "EfficientNetB0": ("models/efficientnetb0_best_94_38.keras", None),
}

test_ds = keras.utils.image_dataset_from_directory(
    "brain_mri/Testing",
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    color_mode="rgb",
    image_size=(224, 224),
    batch_size=32,
    shuffle=False,
)
y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])
y_bin = label_binarize(y_true, classes=range(len(CLASS_NAMES)))

fpr_grid = np.linspace(0, 1, 501)
rows = []
macro_curves = {}

for name, (path, preprocess_fn) in MODELS.items():
    custom = {"preprocess_input": preprocess_fn} if preprocess_fn else None
    model = keras.models.load_model(path, safe_mode=False, custom_objects=custom)
    probs = model.predict(test_ds, verbose=0)
    keras.backend.clear_session()

    fig, ax = plt.subplots(figsize=(6, 5))
    tprs = []
    row = {"model": name}
    for i, c in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        class_auc = auc(fpr, tpr)
        row[c] = class_auc
        tprs.append(np.interp(fpr_grid, fpr, tpr))
        ax.plot(fpr, tpr, label=f"{c} (AUC = {class_auc:.4f})")

    macro_auc = roc_auc_score(y_bin, probs, average="macro")
    row["macro"] = macro_auc
    rows.append(row)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    macro_curves[name] = (mean_tpr, macro_auc)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{name} - ROC (one-vs-rest)\nmacro AUC = {macro_auc:.4f}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_roc.png", dpi=150)
    plt.close(fig)
    print(f"{name}: macro AUC = {macro_auc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, zoom in zip(axes, [False, True]):
    for name, (mean_tpr, macro_auc) in macro_curves.items():
        ax.plot(fpr_grid, mean_tpr, label=f"{name} (AUC = {macro_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    if zoom:
        ax.set_xlim(0, 0.2)
        ax.set_ylim(0.8, 1.0)
        ax.set_title("Macro-average ROC (zoomed)")
    else:
        ax.set_title("Macro-average ROC")
    ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT_DIR / "macro_roc_comparison.png", dpi=150)
plt.close(fig)

table = pd.DataFrame(rows)
table.to_csv(OUT_DIR / "auc_table.csv", index=False)
print("\n", table.round(4).to_string(index=False))
print(f"\nSaved in {OUT_DIR}")