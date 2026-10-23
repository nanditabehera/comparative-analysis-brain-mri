import os
import shutil
import random
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "brain_mri"

TRAIN_DIR = DATA_DIR / "Training"
VALID_DIR = DATA_DIR / "Validation"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
VALIDATION_SPLIT = 0.15
random.seed(42)

# Check that the training folders exist
if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

# Create validation folders and copy a random subset of images
for class_name in CLASSES:
    source_dir = TRAIN_DIR / class_name
    target_dir = VALID_DIR / class_name

    if not source_dir.exists():
        raise FileNotFoundError(f"Class folder not found: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    images = [
        file for file in source_dir.iterdir()
        if file.is_file() and file.suffix.lower() in
        {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    random.shuffle(images)
    count = int(len(images) * VALIDATION_SPLIT)

    for image_file in images[:count]:
      shutil.move(str(image_file), str(target_dir / image_file.name))

    print(
        f"{class_name}: {len(images)} training images, "
        f"{count} validation images"
    )

print("\nValidation split created successfully!")