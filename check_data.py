from pathlib import Path

BASE_DIR = Path(__file__).parent / "brain_mri"

classes = ["glioma", "meningioma", "notumor", "pituitary"]

for c in classes:
    train_path = BASE_DIR / "Training" / c
    val_path = BASE_DIR / "Validation" / c
    test_path = BASE_DIR / "Testing" / c

    train_count = len(list(train_path.glob("*")))
    val_count = len(list(val_path.glob("*")))
    test_count = len(list(test_path.glob("*")))

    print(
        f"{c}: Training = {train_count}, "
        f"Validation = {val_count}, Testing = {test_count}"
    )