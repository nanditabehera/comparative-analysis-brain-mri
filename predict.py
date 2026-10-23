import numpy as np
import keras

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_PATH = r"brain_mri/Testing/pituitary/Te-pi_2.jpg"   # change this

model = keras.models.load_model("models/mobilenetv2_best_96.keras")

img = keras.utils.load_img(IMAGE_PATH, color_mode="rgb", target_size=(224, 224))
x = np.expand_dims(keras.utils.img_to_array(img), axis=0)   # raw 0-255, the model rescales itself

probs = model.predict(x, verbose=0)[0]
for name, p in zip(CLASS_NAMES, probs):
    print(f"{name}: {p * 100:.2f}%")
print("Prediction:", CLASS_NAMES[int(np.argmax(probs))])