from pathlib import Path

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/mobilenetv2_best_96.keras"
IMG_PATH = r"brain_mri/Testing/pituitary/Te-pi_4.jpg"   
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
OUT_DIR = Path("results/gradcam")
OUT_DIR.mkdir(parents=True, exist_ok=True)

model = keras.models.load_model(MODEL_PATH, compile=False)

# model = input -> rescaling -> MobileNetV2 base -> pooling -> dropout -> dense
base = next(l for l in model.layers if isinstance(l, keras.Model))
assert base.layers[-1].name == "out_relu", "unexpected base model"
idx = model.layers.index(base)
pre_layers = model.layers[1:idx]      # rescaling
post_layers = model.layers[idx + 1:]  # pooling, dropout, dense

# Image: raw 0-255, the model rescales itself
img = keras.utils.load_img(IMG_PATH, color_mode="rgb", target_size=(224, 224))
img_array = keras.utils.img_to_array(img)
original = img_array.astype("uint8")
input_tensor = tf.convert_to_tensor(np.expand_dims(img_array, 0), dtype=tf.float32)

preds = model.predict(input_tensor, verbose=0)[0]
pred_idx = int(np.argmax(preds))
pred_name = CLASS_NAMES[pred_idx]
confidence = preds[pred_idx] * 100
true_name = Path(IMG_PATH).parent.name

print(f"True class : {true_name}")
print(f"Prediction : {pred_name} ({confidence:.2f}%)")
for name, p in zip(CLASS_NAMES, preds):
    print(f"  {name:12s}: {p * 100:.2f}%")

# Grad-CAM on the last conv feature map (7x7x1280)
with tf.GradientTape() as tape:
    x = input_tensor
    for layer in pre_layers:
        x = layer(x)
    conv_out = base(x, training=False)
    tape.watch(conv_out)
    x = conv_out
    for layer in post_layers:
        x = layer(x, training=False)
    score = x[:, pred_idx]

grads = tape.gradient(score, conv_out)
weights = tf.reduce_mean(grads, axis=(0, 1, 2))
heatmap = tf.reduce_sum(conv_out[0] * weights, axis=-1).numpy()
heatmap = np.maximum(heatmap, 0)
heatmap = heatmap / (heatmap.max() + 1e-8)
heatmap = cv2.resize(heatmap, (224, 224))

# Overlay
heat_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
orig_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
overlay = cv2.addWeighted(orig_bgr, 0.6, heat_color, 0.4, 0)
overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

stem = Path(IMG_PATH).stem
cv2.imwrite(str(OUT_DIR / f"{stem}_overlay.jpg"), overlay)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original)
axes[0].set_title(f"Original (true: {true_name})")
axes[1].imshow(heatmap, cmap="jet")
axes[1].set_title("Grad-CAM heatmap")
axes[2].imshow(overlay_rgb)
axes[2].set_title(f"Overlay: {pred_name} ({confidence:.2f}%)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / f"{stem}_plot.jpg", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nSaved in {OUT_DIR}: {stem}_overlay.jpg, {stem}_plot.jpg")