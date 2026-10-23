# Brain MRI Tumor Classification with Grad-CAM

Compares four pretrained CNNs (transfer learning) for classifying brain MRI images into four classes, and uses Grad-CAM to see which image regions drive each prediction.

**Classes:** glioma, meningioma, notumor, pituitary

**Models:** MobileNetV2, ResNet50, DenseNet121, EfficientNetB0

> Research and learning project. Not a medical device and not for clinical use.

## Results (Testing set, 1,600 images, 400 per class)

| Model | Test accuracy | Glioma recall | Meningioma recall | Notumor recall | Pituitary recall |
|---|---|---|---|---|---|
| EfficientNetB0 | 94.38% | 82.5% | 95.25% | 99.75% | 100% |
| MobileNetV2 | 94.19% | 82.5% | 95.0% | 99.75% | 99.5% |
| ResNet50 | 94.06% | 79.0% | 97.75% | 100% | 99.5% |
| DenseNet121 | 91.25% | 76.75% | 89.25% | 99.75% | 99.25% |

## Method

- Data: 4,760 training, 840 validation, 1,600 testing images. Validation was cut from the original training set (15%).
- Input: 224x224 RGB. Each model uses its own ImageNet preprocessing, built into the saved model.
- Augmentation (training only): small rotation, zoom and brightness change.
- Head: global average pooling, dropout 0.3, 4-class softmax.
- Phase 1: base frozen, train the head (up to 5 epochs, learning rate 1e-3).
- Phase 2: unfreeze the top 100 base layers, BatchNorm layers stay frozen (up to 25 epochs, learning rate 5e-5).
- Callbacks: best model by validation loss, early stopping (patience 6), reduce learning rate on plateau (patience 3).
- Settings are identical for all four models, so differences come from the architecture.
- The Testing set is used only for final evaluation, never for choosing models or settings.
- Trained on CPU only (Windows, no GPU).

## Limitations

- 36 validation images (about 4%) have exact duplicates in the training set. Exact duplicates between Testing and the other splits: none found. Within-split duplicates exist and are harmless for leakage.
- Splits are by image, not by patient. Near-duplicate images and same-patient slices were not checked.
- Grad-CAM shows where the model looked, not that the region is a real tumor. Heatmaps are coarse (7x7 grid stretched to 224x224) and scaled so the brightest spot is always 1.
- Grad-CAM is computed from the score before softmax, because softmax saturation on very confident predictions gave unreliable maps.
- One training run per model, one seed. No confidence intervals.

## Setup

Python 3.13, TensorFlow 2.21, Keras 3.15 (see `requirements.txt`).

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Not included in this repo. Download it from: <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>

Expected layout:

```text
brain_mri/
  Training/    glioma/ meningioma/ notumor/ pituitary/
  Validation/  glioma/ meningioma/ notumor/ pituitary/
  Testing/     glioma/ meningioma/ notumor/ pituitary/
```

`prepare_data.py` creates `Validation/` by moving 15% of the training images. `check_data.py` prints image counts per split. `check_duplicates.py` finds exact duplicate files across splits.

## Run

```powershell
python mobilenet_model.py
python resnet_model.py
python densenet_model.py
python efficientnet_model.py
```

Each script saves `models/<name>_best.keras`, a history CSV and a curves plot in `results/training_curves/`.

Evaluate one model on the Testing set: set `NAME` at the top of `evaluate_models.py`, then

```powershell
python evaluate_models.py
```

ROC curves and AUC table (all four models): `python roc_auc.py` (saves to `results/roc/`).

Grad-CAM: put image paths in `IMAGE_PATHS` in `gradcam.py`, then `python gradcam.py` (saves to `results/gradcam/`).

ResNet50 and DenseNet121 contain a Lambda preprocessing layer, so loading them needs `safe_mode=False` and `custom_objects={"preprocess_input": ...}` with the matching function. The evaluation and Grad-CAM scripts already do this.

Trained models (`*.keras`) are not in the repo because of file size. <ADD LINK IF YOU SHARE THEM>

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Create the validation split |
| `check_data.py` | Count images per split |
| `check_duplicates.py` | Find exact duplicates across splits |
| `*_model.py` | Train one model (4 scripts) |
| `evaluate_models.py` | Test-set accuracy, report, confusion matrix |
| `roc_auc.py` | ROC curves and AUC |
| `gradcam.py` | Grad-CAM heatmaps |
