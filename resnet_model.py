from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras import layers

# ---------- settings ----------
NAME = "resnet50"                                   # CHANGED
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "brain_mri"
MODELS_DIR = BASE_DIR / "models"
CURVES_DIR = BASE_DIR / "results" / "training_curves"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS_HEAD = 5        # phase 1: base frozen
EPOCHS_FINE = 25       # phase 2: fine-tuning
FINE_TUNE_LAYERS = 100
# ------------------------------

keras.utils.set_random_seed(SEED)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CURVES_DIR.mkdir(parents=True, exist_ok=True)


def load_split(split, shuffle):
    return keras.utils.image_dataset_from_directory(
        str(DATA_DIR / split),
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


# ---------- data ----------
train_ds = load_split("Training", shuffle=True)
val_ds = load_split("Validation", shuffle=False)

augment = keras.Sequential([
    layers.RandomRotation(0.04),    # about +/- 14 degrees
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
])
train_ds = train_ds.map(
    lambda x, y: (augment(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE,
).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# ---------- model ----------
base = keras.applications.ResNet50(                 # CHANGED
    include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,)
)
base.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
# CHANGED: ResNet50 wants BGR order with the ImageNet mean subtracted
x = layers.Lambda(keras.applications.resnet50.preprocess_input,
                  name="resnet_preprocess")(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
model = keras.Model(inputs, outputs, name=NAME)

ckpt_path = str(MODELS_DIR / f"{NAME}_best.keras")


def make_callbacks(threshold=None):
    return [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss", save_best_only=True,
            initial_value_threshold=threshold, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7
        ),
    ]


def compile_model(lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


# ---------- phase 1: train the new head ----------
print("\n=== Phase 1: base frozen ===")
compile_model(1e-3)
h1 = model.fit(train_ds, validation_data=val_ds,
               epochs=EPOCHS_HEAD, callbacks=make_callbacks())
best_phase1 = min(h1.history["val_loss"])

# ---------- phase 2: fine-tune top layers ----------
print("\n=== Phase 2: fine-tuning ===")
base.trainable = True
for layer in base.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False
for layer in base.layers[-FINE_TUNE_LAYERS:]:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
compile_model(5e-5)  # must recompile after changing trainable
h2 = model.fit(train_ds, validation_data=val_ds,
               epochs=EPOCHS_FINE, callbacks=make_callbacks(threshold=best_phase1))

# ---------- save history and curves ----------
df1 = pd.DataFrame(h1.history)
df1["phase"] = 1
df2 = pd.DataFrame(h2.history)
df2["phase"] = 2
df = pd.concat([df1, df2], ignore_index=True)
df.insert(0, "epoch", range(1, len(df) + 1))
df.to_csv(CURVES_DIR / f"{NAME}_history.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, metric in zip(axes, ["loss", "accuracy"]):
    ax.plot(df["epoch"], df[metric], label="train")
    ax.plot(df["epoch"], df["val_" + metric], label="validation")
    ax.axvline(len(df1) + 0.5, color="gray", linestyle="--", label="fine-tuning starts")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"{NAME} - {metric}")
    ax.legend()
fig.tight_layout()
fig.savefig(CURVES_DIR / f"{NAME}_curves.png", dpi=150)
plt.close(fig)

# ---------- final check on the saved best model ----------
best = keras.models.load_model(ckpt_path, safe_mode=False)
loss, acc = best.evaluate(val_ds, verbose=0)
print(f"\nBest saved model: val_loss = {loss:.4f}, val_accuracy = {acc:.4f}")
print(f"Model saved to: {ckpt_path}") 