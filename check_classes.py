import os
import pandas as pd

ROOT = "."
splits = ["train", "valid", "test"]

all_classes = set()

for split in splits:
    csv_path = os.path.join(ROOT, split, "_annotations.csv")

    if not os.path.exists(csv_path):
        print(f"CSV পাওয়া যায়নি: {csv_path}")
        continue

    df = pd.read_csv(
        csv_path,
        header=None,
        names=["filename", "xmin", "ymin", "xmax", "ymax", "class"]
    )

    classes = sorted(
        df["class"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    print(f"\n{split} classes:")
    print(classes)

    all_classes.update(classes)

print("\n========================")
print("All Classes")
print("========================")

for class_id, class_name in enumerate(sorted(all_classes)):
    print(f"{class_id}: {class_name}")