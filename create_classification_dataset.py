from pathlib import Path
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = ROOT / "classification_dataset"


SPLITS = {
    "train": ROOT / "train",
    "valid": ROOT / "valid",
    "test": ROOT / "test",
}


CLASSES = [
    "Cut",
    "Hole",
    "Stain",
    "ThreadError"
]


PADDING = 0.10


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

def create_output():

    if OUTPUT_DIR.exists():
        print("Removing old classification dataset...")
        shutil.rmtree(OUTPUT_DIR)


    for split in SPLITS:

        for cls in CLASSES:

            folder = (
                OUTPUT_DIR /
                split /
                cls
            )

            folder.mkdir(
                parents=True,
                exist_ok=True
            )



# ============================================================
# FIND IMAGE FILE
# ============================================================

def find_image(folder, filename):

    path = folder / filename


    if path.exists():
        return path


    stem = Path(filename).stem


    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG"
    ]


    for ext in extensions:

        candidate = folder / (stem + ext)

        if candidate.exists():

            return candidate


    return None



# ============================================================
# CROP BOX WITH PADDING
# ============================================================

def get_crop_box(
        xmin,
        ymin,
        xmax,
        ymax,
        width,
        height
):

    box_width = xmax - xmin
    box_height = ymax - ymin


    pad_x = box_width * PADDING
    pad_y = box_height * PADDING


    xmin = max(
        0,
        int(xmin - pad_x)
    )

    ymin = max(
        0,
        int(ymin - pad_y)
    )


    xmax = min(
        width,
        int(xmax + pad_x)
    )

    ymax = min(
        height,
        int(ymax + pad_y)
    )


    return (
        xmin,
        ymin,
        xmax,
        ymax
    )



# ============================================================
# PROCESS DATA
# ============================================================

def process_split(split_name):


    folder = SPLITS[split_name]


    csv_file = folder / "_annotations.csv"


    print("\n")
    print("="*60)
    print("Processing:", split_name)
    print("="*60)



    if not csv_file.exists():

        raise FileNotFoundError(
            f"CSV not found: {csv_file}"
        )


    # IMPORTANT:
    # Your CSV has NO header
    df = pd.read_csv(
        csv_file,
        header=None,
        names=[
            "filename",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "class"
        ]
    )


    print("CSV loaded:")
    print(df.head())



    counts = {
        cls:0
        for cls in CLASSES
    }



    skipped = 0



    for index,row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=split_name
    ):


        filename = str(
            row["filename"]
        ).strip()


        class_name = str(
            row["class"]
        ).strip()



        if class_name not in CLASSES:

            skipped += 1
            continue



        image_path = find_image(
            folder,
            filename
        )



        if image_path is None:

            skipped += 1
            continue



        try:

            xmin = int(row["xmin"])
            ymin = int(row["ymin"])
            xmax = int(row["xmax"])
            ymax = int(row["ymax"])

        except:

            skipped += 1
            continue



        image = Image.open(
            image_path
        ).convert("RGB")



        crop_box = get_crop_box(
            xmin,
            ymin,
            xmax,
            ymax,
            image.width,
            image.height
        )


        crop = image.crop(
            crop_box
        )


        # ignore invalid crop
        if crop.width < 5 or crop.height < 5:

            skipped += 1
            continue



        counts[class_name] += 1



        save_name = (
            f"{Path(filename).stem}_"
            f"{class_name}_"
            f"{counts[class_name]:05d}.jpg"
        )



        save_path = (
            OUTPUT_DIR /
            split_name /
            class_name /
            save_name
        )


        crop.save(
            save_path,
            quality=95
        )



    print("\nSummary:", split_name)

    total = 0

    for cls,count in counts.items():

        print(
            f"{cls:<15}: {count}"
        )

        total += count


    print(
        "Total crops:",
        total
    )

    print(
        "Skipped:",
        skipped
    )



# ============================================================
# MAIN
# ============================================================

def main():


    print("="*60)
    print(
        "OBJECT DETECTION TO CLASSIFICATION DATASET"
    )
    print("="*60)



    create_output()



    for split in SPLITS:

        process_split(split)



    print("\n")
    print("="*60)
    print("DONE")
    print("="*60)

    print(
        "Saved at:"
    )

    print(
        OUTPUT_DIR
    )



if __name__ == "__main__":

    main()