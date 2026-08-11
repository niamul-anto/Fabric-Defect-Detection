import os
import shutil
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

# Script যদি dataset root folder-এর ভেতরে থাকে
ROOT = Path(".")

# Original RetinaNet dataset folders
SPLITS = ["train", "valid", "test"]

# YOLO output folder
OUTPUT_ROOT = ROOT / "yolo_dataset"

# Dataset classes
CLASS_MAP = {
    "Cut": 0,
    "Hole": 1,
    "Stain": 2,
    "ThreadError": 3,
}

# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_image(source_folder: Path, filename: str) -> Path | None:
    """
    CSV-তে থাকা filename অনুযায়ী image খুঁজে বের করে।
    """

    direct_path = source_folder / filename

    if direct_path.exists():
        return direct_path

    target_name = filename.lower()

    for item in source_folder.iterdir():
        if item.is_file() and item.name.lower() == target_name:
            return item

    return None


def normalize_box(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """
    RetinaNet pixel coordinates কে YOLO normalized format-এ convert করে।
    """

    xmin = max(0.0, min(xmin, float(image_width)))
    xmax = max(0.0, min(xmax, float(image_width)))
    ymin = max(0.0, min(ymin, float(image_height)))
    ymax = max(0.0, min(ymax, float(image_height)))

    if xmax <= xmin or ymax <= ymin:
        return None

    x_center = ((xmin + xmax) / 2.0) / image_width
    y_center = ((ymin + ymax) / 2.0) / image_height
    box_width = (xmax - xmin) / image_width
    box_height = (ymax - ymin) / image_height

    values = (x_center, y_center, box_width, box_height)

    if not all(0.0 <= value <= 1.0 for value in values):
        return None

    return values


def read_annotations(csv_path: Path) -> pd.DataFrame:
    """
    Header ছাড়া RetinaNet CSV পড়ে।

    Expected format:
    filename,xmin,ymin,xmax,ymax,class
    """

    columns = [
        "filename",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "class",
    ]

    dataframe = pd.read_csv(
        csv_path,
        header=None,
        names=columns,
    )

    dataframe["filename"] = (
        dataframe["filename"]
        .astype(str)
        .str.strip()
    )

    dataframe["class"] = (
        dataframe["class"]
        .astype(str)
        .str.strip()
    )

    for column in ["xmin", "ymin", "xmax", "ymax"]:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = dataframe.dropna(
        subset=[
            "filename",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "class",
        ]
    )

    return dataframe


def clear_folder(folder: Path) -> None:
    """
    Output images এবং labels folder পরিষ্কার করে।
    """

    if folder.exists():
        shutil.rmtree(folder)

    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# SPLIT CONVERSION
# ============================================================

def convert_split(split: str) -> dict:
    """
    একটি split RetinaNet CSV format থেকে YOLO format-এ convert করে।
    """

    source_folder = ROOT / split
    csv_path = source_folder / "_annotations.csv"

    output_split_folder = OUTPUT_ROOT / split
    output_images_folder = output_split_folder / "images"
    output_labels_folder = output_split_folder / "labels"

    print("\n" + "=" * 60)
    print(f"Processing split: {split}")
    print("=" * 60)

    if not source_folder.exists():
        raise FileNotFoundError(
            f"Source folder পাওয়া যায়নি: {source_folder.resolve()}"
        )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV পাওয়া যায়নি: {csv_path.resolve()}"
        )

    clear_folder(output_images_folder)
    clear_folder(output_labels_folder)

    dataframe = read_annotations(csv_path)

    print(f"Total annotation rows: {len(dataframe)}")

    unknown_classes = sorted(
        set(dataframe["class"].unique()) - set(CLASS_MAP.keys())
    )

    if unknown_classes:
        raise ValueError(
            f"Unknown classes পাওয়া গেছে: {unknown_classes}\n"
            f"CLASS_MAP update করুন।"
        )

    grouped_annotations = dataframe.groupby("filename")

    copied_images = 0
    created_labels = 0
    valid_boxes = 0
    skipped_boxes = 0
    missing_images = 0

    processed_image_names = set()

    # --------------------------------------------------------
    # Images with annotations
    # --------------------------------------------------------

    for filename, rows in grouped_annotations:
        source_image_path = find_image(source_folder, filename)

        if source_image_path is None:
            print(f"[Missing image] {filename}")
            missing_images += 1
            continue

        try:
            with Image.open(source_image_path) as image:
                image_width, image_height = image.size
        except Exception as error:
            print(f"[Cannot open image] {filename}: {error}")
            missing_images += 1
            continue

        destination_image_path = (
            output_images_folder / source_image_path.name
        )

        shutil.copy2(
            source_image_path,
            destination_image_path,
        )

        copied_images += 1
        processed_image_names.add(source_image_path.name.lower())

        label_path = (
            output_labels_folder
            / f"{source_image_path.stem}.txt"
        )

        label_lines = []

        for _, row in rows.iterrows():
            class_name = row["class"]
            class_id = CLASS_MAP[class_name]

            converted_box = normalize_box(
                xmin=float(row["xmin"]),
                ymin=float(row["ymin"]),
                xmax=float(row["xmax"]),
                ymax=float(row["ymax"]),
                image_width=image_width,
                image_height=image_height,
            )

            if converted_box is None:
                print(
                    f"[Invalid box] {filename} | "
                    f"{row['xmin']}, {row['ymin']}, "
                    f"{row['xmax']}, {row['ymax']}"
                )
                skipped_boxes += 1
                continue

            x_center, y_center, box_width, box_height = converted_box

            label_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )

            valid_boxes += 1

        with open(
            label_path,
            "w",
            encoding="utf-8",
        ) as label_file:
            if label_lines:
                label_file.write("\n".join(label_lines) + "\n")

        created_labels += 1

    # --------------------------------------------------------
    # Images without annotations
    # --------------------------------------------------------

    for item in source_folder.iterdir():
        if not item.is_file():
            continue

        if item.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if item.name.lower() in processed_image_names:
            continue

        destination_image_path = output_images_folder / item.name
        shutil.copy2(item, destination_image_path)

        empty_label_path = (
            output_labels_folder / f"{item.stem}.txt"
        )

        empty_label_path.write_text(
            "",
            encoding="utf-8",
        )

        copied_images += 1
        created_labels += 1

    print(f"Copied images: {copied_images}")
    print(f"Created label files: {created_labels}")
    print(f"Valid bounding boxes: {valid_boxes}")
    print(f"Skipped bounding boxes: {skipped_boxes}")
    print(f"Missing images: {missing_images}")

    return {
        "split": split,
        "images": copied_images,
        "labels": created_labels,
        "valid_boxes": valid_boxes,
        "skipped_boxes": skipped_boxes,
        "missing_images": missing_images,
    }


# ============================================================
# DATA.YAML CREATION
# ============================================================

def create_data_yaml() -> None:
    """
    Ultralytics YOLO-এর জন্য data.yaml তৈরি করে।
    """

    ordered_names = [
        class_name
        for class_name, class_id in sorted(
            CLASS_MAP.items(),
            key=lambda item: item[1],
        )
    ]

    yaml_content = {
        "path": str(OUTPUT_ROOT.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {
            index: class_name
            for index, class_name in enumerate(ordered_names)
        },
    }

    yaml_path = OUTPUT_ROOT / "data.yaml"

    with open(
        yaml_path,
        "w",
        encoding="utf-8",
    ) as yaml_file:
        yaml.safe_dump(
            yaml_content,
            yaml_file,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"\ndata.yaml created: {yaml_path.resolve()}")


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_output() -> None:
    """
    Output dataset-এর image-label matching check করে।
    """

    print("\n" + "=" * 60)
    print("Validating converted YOLO dataset")
    print("=" * 60)

    total_images = 0
    total_labels = 0
    total_missing_labels = 0
    total_orphan_labels = 0

    for split in SPLITS:
        images_folder = OUTPUT_ROOT / split / "images"
        labels_folder = OUTPUT_ROOT / split / "labels"

        image_files = {
            item.stem
            for item in images_folder.iterdir()
            if item.is_file()
            and item.suffix.lower() in IMAGE_EXTENSIONS
        }

        label_files = {
            item.stem
            for item in labels_folder.glob("*.txt")
        }

        missing_labels = image_files - label_files
        orphan_labels = label_files - image_files

        total_images += len(image_files)
        total_labels += len(label_files)
        total_missing_labels += len(missing_labels)
        total_orphan_labels += len(orphan_labels)

        print(f"\n{split}:")
        print(f"  Images: {len(image_files)}")
        print(f"  Labels: {len(label_files)}")
        print(f"  Missing labels: {len(missing_labels)}")
        print(f"  Orphan labels: {len(orphan_labels)}")

        if missing_labels:
            print(
                "  Missing label examples:",
                sorted(missing_labels)[:5],
            )

        if orphan_labels:
            print(
                "  Orphan label examples:",
                sorted(orphan_labels)[:5],
            )

    print("\nOverall validation:")
    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")
    print(f"Total missing labels: {total_missing_labels}")
    print(f"Total orphan labels: {total_orphan_labels}")

    if total_missing_labels == 0 and total_orphan_labels == 0:
        print("\nValidation successful.")
    else:
        print("\nValidation warning: কিছু image-label mismatch আছে।")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 60)
    print("RetinaNet CSV to YOLO Dataset Converter")
    print("=" * 60)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    for split in SPLITS:
        summary = convert_split(split)
        summaries.append(summary)

    create_data_yaml()
    validate_output()

    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)

    for summary in summaries:
        print(
            f"{summary['split']}: "
            f"images={summary['images']}, "
            f"labels={summary['labels']}, "
            f"boxes={summary['valid_boxes']}, "
            f"skipped={summary['skipped_boxes']}, "
            f"missing={summary['missing_images']}"
        )

    print("\nYOLO dataset তৈরি হয়েছে:")
    print(OUTPUT_ROOT.resolve())

    print("\nExpected structure:")

    print(
        """
yolo_dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
"""
    )


if __name__ == "__main__":
    main()