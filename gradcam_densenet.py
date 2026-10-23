from pathlib import Path

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_PATH = "models/densenet121_best_91_25.keras"
OUT_DIR = Path("results/gradcam/densenet121")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_PATHS = [
    "brain_mri/Testing/glioma/Te-gl_11.jpg",
]

model = keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False,
    custom_objects={"preprocess_input": keras.applications.densenet.preprocess_input},
)

base = next(l for l in model.layers if isinstance(l, keras.Model))
i = model.layers.index(base)
pre_layers = model.layers[1:i]
post_layers = model.layers[i + 1:]
print("Last conv layer used:", base.layers[-1].name)


def load_image(path):
    img = keras.utils.load_img(path, color_mode="rgb", target_size=(224, 224))
    return keras.utils.img_to_array(img)


def gradcam(arr):
    with tf.GradientTape() as tape:
        x = tf.convert_to_tensor(arr[None], dtype=tf.float32)
        for l in pre_layers:
            x = l(x)
        conv = base(x, training=False)
        tape.watch(conv)
        x = conv
        for l in post_layers[:-1]:
            x = l(x, training=False)
        dense = post_layers[-1]
        logits = tf.matmul(x, dense.kernel) + dense.bias
        probs = tf.nn.softmax(logits)[0]
        idx = int(tf.argmax(logits[0]))
        score = logits[:, idx]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv[0] * weights, axis=-1).numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cv2.resize(cam, (224, 224))
    return cam, idx, probs.numpy()


for p in map(Path, IMAGE_PATHS):
    arr = load_image(p)
    true_name = p.parent.name
    cam, idx, probs = gradcam(arr)
    pred_name = CLASS_NAMES[idx]
    conf = probs[idx] * 100

    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    orig_bgr = cv2.cvtColor(arr.astype("uint8"), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(orig_bgr, 0.6, heat, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    print(f"\n{p.name} (true: {true_name})")
    print(f"  Prediction: {pred_name} ({conf:.2f}%)")
    for name, pr in zip(CLASS_NAMES, probs):
        print(f"    {name:12s}: {pr * 100:.2f}%")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(arr.astype("uint8"))
    axes[0].set_title(f"Original (true: {true_name})")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM heatmap")
    axes[2].imshow(overlay_rgb)
    axes[2].set_title(f"Overlay: {pred_name} ({conf:.1f}%)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{p.stem}_densenet121.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"\nSaved in {OUT_DIR}")