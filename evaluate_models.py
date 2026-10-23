import numpy as np
import keras
from sklearn.metrics import classification_report, confusion_matrix


NAME = "densenet121"   # "mobilenetv2", "resnet50", "densenet121" or "efficientnetb0"


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# model file, and the preprocessing function its Lambda layer needs (None if it has none)
MODELS = {
    "mobilenetv2":    ("models/mobilenetv2_best_94_19.keras", None),
    "resnet50":       ("models/resnet50_best_94_06.keras",
                       keras.applications.resnet50.preprocess_input),
    "densenet121":    ("models/densenet121_best_91_25.keras",
                       keras.applications.densenet.preprocess_input),
    "efficientnetb0": ("models/efficientnetb0_best_94_38.keras", None),
}

path, preprocess_fn = MODELS[NAME]
custom = {"preprocess_input": preprocess_fn} if preprocess_fn else None

# safe_mode=False is only needed for models with a Lambda layer. These are your own files.
model = keras.models.load_model(path, safe_mode=False, custom_objects=custom)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print(f"Evaluating: {NAME} ({path})")

# Test set (no shuffle, no augmentation)
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

# Evaluate
loss, accuracy = model.evaluate(test_ds, verbose=1)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Predictions
y_pred = np.argmax(model.predict(test_ds, verbose=1), axis=1)
y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

print("Confusion Matrix (rows = true, columns = predicted):")
print(confusion_matrix(y_true, y_pred))