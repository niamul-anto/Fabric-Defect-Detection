import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import csv
import gc

import torch
from PIL import Image
from torchvision.ops import box_iou
from tqdm import tqdm
from ultralytics import YOLO


# ============================================================
# PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATASET_ROOT = ROOT / "yolo_dataset"
DATA_YAML = DATASET_ROOT / "data.yaml"

MODEL_PATH = (
    ROOT
    / "results"
    / "yolov8m_fabric_960"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "yolov8m_fabric_960"
    / "threshold_evaluation"
)

CLASS_NAMES = [
    "Cut",
    "Hole",
    "Stain",
    "ThreadError",
]

NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 960
MAX_DETECTIONS = 300

# Validation set-এ এগুলো পরীক্ষা করা হবে
CONFIDENCE_THRESHOLDS = [
    0.50,
    0.45,
    0.40,
    0.35,
    0.30,
    0.25,
    0.20,
    0.15,
    0.10,
    0.05,
]

# Prediction এবং ground-truth match করার IoU
EVALUATION_IOU_THRESHOLD = 0.50

# YOLO Non-Maximum Suppression threshold
NMS_IOU_THRESHOLD = 0.70

# AP curve তৈরির জন্য খুব কম confidence
MINIMUM_PREDICTION_CONFIDENCE = 0.001

DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_HALF_PRECISION = torch.cuda.is_available()


# ============================================================
# CUDA MEMORY CLEANUP
# ============================================================

def clear_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# READ YOLO LABELS
# ============================================================

def read_yolo_labels(
    label_path: Path,
    image_width: int,
    image_height: int,
):
    boxes = []
    labels = []

    if not label_path.exists():
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )

    text = label_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )

    for line in text.splitlines():
        values = line.strip().split()

        if len(values) != 5:
            print(
                f"Warning: invalid annotation ignored: "
                f"{label_path.name}: {line}"
            )
            continue

        try:
            (
                class_id,
                x_center,
                y_center,
                box_width,
                box_height,
            ) = map(float, values)

        except ValueError:
            print(
                f"Warning: non-numeric annotation ignored: "
                f"{label_path.name}: {line}"
            )
            continue

        class_id = int(class_id)

        if class_id < 0 or class_id >= NUM_CLASSES:
            print(
                f"Warning: invalid class ID ignored: "
                f"{label_path.name}: {class_id}"
            )
            continue

        # YOLO normalized coordinates থেকে pixel coordinates
        x_center *= image_width
        y_center *= image_height
        box_width *= image_width
        box_height *= image_height

        xmin = x_center - box_width / 2
        ymin = y_center - box_height / 2
        xmax = x_center + box_width / 2
        ymax = y_center + box_height / 2

        xmin = max(0.0, min(xmin, float(image_width)))
        ymin = max(0.0, min(ymin, float(image_height)))
        xmax = max(0.0, min(xmax, float(image_width)))
        ymax = max(0.0, min(ymax, float(image_height)))

        if xmax <= xmin or ymax <= ymin:
            print(
                f"Warning: invalid bounding box ignored: "
                f"{label_path.name}: {line}"
            )
            continue

        boxes.append([
            xmin,
            ymin,
            xmax,
            ymax,
        ])

        labels.append(class_id)

    if not boxes:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )

    return (
        torch.tensor(
            boxes,
            dtype=torch.float32,
        ),
        torch.tensor(
            labels,
            dtype=torch.int64,
        ),
    )


# ============================================================
# LOAD SPLIT INFORMATION
# ============================================================

def load_split_information(split_name: str):
    image_dir = (
        DATASET_ROOT
        / split_name
        / "images"
    )

    label_dir = (
        DATASET_ROOT
        / split_name
        / "labels"
    )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image folder পাওয়া যায়নি:\n{image_dir}"
        )

    if not label_dir.exists():
        raise FileNotFoundError(
            f"Label folder পাওয়া যায়নি:\n{label_dir}"
        )

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in supported_extensions
    )

    if not image_paths:
        raise RuntimeError(
            f"No images found in:\n{image_dir}"
        )

    dataset_information = []

    total_ground_truth = 0

    for image_path in image_paths:
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        gt_boxes, gt_labels = read_yolo_labels(
            label_path=label_path,
            image_width=image_width,
            image_height=image_height,
        )

        total_ground_truth += len(gt_boxes)

        dataset_information.append({
            "image_path": image_path,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
        })

    print(
        f"{split_name.capitalize()} images       : "
        f"{len(dataset_information)}"
    )

    print(
        f"{split_name.capitalize()} instances    : "
        f"{total_ground_truth}"
    )

    return dataset_information


# ============================================================
# GENERATE PREDICTIONS ONE IMAGE AT A TIME
# ============================================================

@torch.inference_mode()
def generate_predictions(
    model: YOLO,
    dataset_information,
    split_name: str,
):
    print(
        f"\nGenerating {split_name} predictions "
        f"one image at a time..."
    )

    saved_predictions = []

    for item in tqdm(
        dataset_information,
        desc=f"Predicting {split_name}",
    ):
        image_path = item["image_path"]

        result = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,

            # সব threshold পরে পরীক্ষা করার জন্য low confidence
            conf=MINIMUM_PREDICTION_CONFIDENCE,

            iou=NMS_IOU_THRESHOLD,
            device=DEVICE,

            # একবারে শুধু একটি image
            batch=1,

            max_det=MAX_DETECTIONS,
            half=USE_HALF_PRECISION,
            verbose=False,
        )[0]

        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):
            pred_boxes = (
                result.boxes.xyxy
                .detach()
                .cpu()
                .float()
            )

            pred_labels = (
                result.boxes.cls
                .detach()
                .cpu()
                .long()
            )

            pred_scores = (
                result.boxes.conf
                .detach()
                .cpu()
                .float()
            )

        else:
            pred_boxes = torch.zeros(
                (0, 4),
                dtype=torch.float32,
            )

            pred_labels = torch.zeros(
                (0,),
                dtype=torch.int64,
            )

            pred_scores = torch.zeros(
                (0,),
                dtype=torch.float32,
            )

        saved_predictions.append({
            "boxes": pred_boxes,
            "labels": pred_labels,
            "scores": pred_scores,
        })

        del result

    clear_memory()

    return saved_predictions


# ============================================================
# MATCH PREDICTIONS WITH GROUND TRUTH
# ============================================================

def evaluate_at_threshold(
    dataset_information,
    saved_predictions,
    confidence_threshold: float,
):
    class_statistics = {
        class_id: {
            "TP": 0,
            "FP": 0,
            "FN": 0,
        }
        for class_id in range(NUM_CLASSES)
    }

    for dataset_item, prediction_item in zip(
        dataset_information,
        saved_predictions,
    ):
        gt_boxes = dataset_item["gt_boxes"]
        gt_labels = dataset_item["gt_labels"]

        all_pred_boxes = prediction_item["boxes"]
        all_pred_labels = prediction_item["labels"]
        all_pred_scores = prediction_item["scores"]

        confidence_mask = (
            all_pred_scores
            >= confidence_threshold
        )

        pred_boxes = all_pred_boxes[
            confidence_mask
        ]

        pred_labels = all_pred_labels[
            confidence_mask
        ]

        pred_scores = all_pred_scores[
            confidence_mask
        ]

        # High-confidence prediction আগে match হবে
        if len(pred_scores) > 0:
            order = torch.argsort(
                pred_scores,
                descending=True,
            )

            pred_boxes = pred_boxes[order]
            pred_labels = pred_labels[order]

        for class_id in range(NUM_CLASSES):
            class_gt_boxes = gt_boxes[
                gt_labels == class_id
            ]

            class_pred_boxes = pred_boxes[
                pred_labels == class_id
            ]

            matched_gt = torch.zeros(
                len(class_gt_boxes),
                dtype=torch.bool,
            )

            class_tp = 0
            class_fp = 0

            for predicted_box in class_pred_boxes:
                if len(class_gt_boxes) == 0:
                    class_fp += 1
                    continue

                ious = box_iou(
                    predicted_box.unsqueeze(0),
                    class_gt_boxes,
                ).squeeze(0)

                # একই ground-truth box দ্বিতীয়বার match হবে না
                ious[matched_gt] = -1.0

                best_iou, best_gt_index = torch.max(
                    ious,
                    dim=0,
                )

                if (
                    float(best_iou)
                    >= EVALUATION_IOU_THRESHOLD
                ):
                    class_tp += 1
                    matched_gt[best_gt_index] = True
                else:
                    class_fp += 1

            class_fn = int(
                (~matched_gt).sum().item()
            )

            class_statistics[class_id]["TP"] += class_tp
            class_statistics[class_id]["FP"] += class_fp
            class_statistics[class_id]["FN"] += class_fn

    total_tp = sum(
        values["TP"]
        for values in class_statistics.values()
    )

    total_fp = sum(
        values["FP"]
        for values in class_statistics.values()
    )

    total_fn = sum(
        values["FN"]
        for values in class_statistics.values()
    )

    precision = (
        total_tp
        / max(total_tp + total_fp, 1)
    )

    recall = (
        total_tp
        / max(total_tp + total_fn, 1)
    )

    f1_score = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

    class_results = []

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        tp = class_statistics[class_id]["TP"]
        fp = class_statistics[class_id]["FP"]
        fn = class_statistics[class_id]["FN"]

        class_precision = (
            tp / max(tp + fp, 1)
        )

        class_recall = (
            tp / max(tp + fn, 1)
        )

        class_f1 = (
            2 * class_precision * class_recall
            / max(
                class_precision + class_recall,
                1e-12,
            )
        )

        class_results.append({
            "Class": class_name,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": class_precision,
            "Recall": class_recall,
            "F1-score": class_f1,
        })

    return {
        "Confidence": confidence_threshold,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_score,
        "Class results": class_results,
    }


# ============================================================
# STANDARD ULTRALYTICS mAP
# ============================================================

def evaluate_standard_map(model: YOLO):
    print("\nCalculating standard YOLO test mAP...")

    clear_memory()

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",

        imgsz=IMAGE_SIZE,
        batch=1,

        device=DEVICE,
        workers=0,

        conf=MINIMUM_PREDICTION_CONFIDENCE,
        iou=NMS_IOU_THRESHOLD,

        max_det=MAX_DETECTIONS,

        plots=False,
        save_json=False,
        verbose=True,
    )

    results = {
        "Ultralytics Precision": float(
            metrics.box.mp
        ),
        "Ultralytics Recall": float(
            metrics.box.mr
        ),
        "mAP@0.5": float(
            metrics.box.map50
        ),
        "mAP@0.5:0.95": float(
            metrics.box.map
        ),
    }

    clear_memory()

    return results


# ============================================================
# PRINT THRESHOLD TABLE
# ============================================================

def print_threshold_table(
    split_name: str,
    results,
):
    print("\n" + "=" * 100)
    print(
        f"{split_name.upper()} CONFIDENCE "
        f"THRESHOLD RESULTS"
    )
    print("=" * 100)

    print(
        f"{'CONF':<10}"
        f"{'TP':<8}"
        f"{'FP':<8}"
        f"{'FN':<8}"
        f"{'PRECISION':<15}"
        f"{'RECALL':<15}"
        f"{'F1-SCORE':<15}"
    )

    print("-" * 100)

    for result in results:
        print(
            f"{result['Confidence']:<10.2f}"
            f"{result['TP']:<8}"
            f"{result['FP']:<8}"
            f"{result['FN']:<8}"
            f"{result['Precision']:<15.6f}"
            f"{result['Recall']:<15.6f}"
            f"{result['F1-score']:<15.6f}"
        )


# ============================================================
# SAVE VALIDATION THRESHOLD RESULTS
# ============================================================

def save_validation_results(validation_results):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "validation_threshold_results.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "Split",
            "Confidence",
            "IoU threshold",
            "TP",
            "FP",
            "FN",
            "Precision",
            "Recall",
            "F1-score",
        ])

        for result in validation_results:
            writer.writerow([
                "valid",
                result["Confidence"],
                EVALUATION_IOU_THRESHOLD,
                result["TP"],
                result["FP"],
                result["FN"],
                result["Precision"],
                result["Recall"],
                result["F1-score"],
            ])

    print(f"Saved: {output_path}")


# ============================================================
# SAVE FINAL TEST RESULTS
# ============================================================

def save_final_test_results(
    best_threshold,
    test_result,
    map_results,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_path = (
        OUTPUT_DIR
        / "yolov8m_final_test_results.csv"
    )

    with open(
        overall_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "Model",
            "Selected confidence",
            "Evaluation IoU",
            "TP",
            "FP",
            "FN",
            "Precision",
            "Recall",
            "F1-score",
            "mAP@0.5",
            "mAP@0.5:0.95",
            "Ultralytics Precision",
            "Ultralytics Recall",
        ])

        writer.writerow([
            "YOLOv8m",
            best_threshold,
            EVALUATION_IOU_THRESHOLD,
            test_result["TP"],
            test_result["FP"],
            test_result["FN"],
            test_result["Precision"],
            test_result["Recall"],
            test_result["F1-score"],
            map_results["mAP@0.5"],
            map_results["mAP@0.5:0.95"],
            map_results["Ultralytics Precision"],
            map_results["Ultralytics Recall"],
        ])

    class_path = (
        OUTPUT_DIR
        / "yolov8m_final_class_results.csv"
    )

    with open(
        class_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "Confidence",
            "Class",
            "TP",
            "FP",
            "FN",
            "Precision",
            "Recall",
            "F1-score",
        ])

        for class_result in test_result[
            "Class results"
        ]:
            writer.writerow([
                best_threshold,
                class_result["Class"],
                class_result["TP"],
                class_result["FP"],
                class_result["FN"],
                class_result["Precision"],
                class_result["Recall"],
                class_result["F1-score"],
            ])

    print(f"Saved: {overall_path}")
    print(f"Saved: {class_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("YOLOv8m Threshold Selection and Final Test Evaluation")
    print("=" * 80)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nbest.pt পাওয়া যায়নি:\n"
            f"{MODEL_PATH}\n"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"\ndata.yaml পাওয়া যায়নি:\n"
            f"{DATA_YAML}\n"
        )

    print(f"Model             : {MODEL_PATH}")
    print(f"Dataset           : {DATA_YAML}")
    print(f"Device            : {DEVICE}")
    print(f"Image size        : {IMAGE_SIZE}")
    print(f"Evaluation IoU    : {EVALUATION_IOU_THRESHOLD}")
    print(f"NMS IoU           : {NMS_IOU_THRESHOLD}")
    print(f"Half precision    : {USE_HALF_PRECISION}")

    if torch.cuda.is_available():
        print(
            f"GPU               : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory        : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    clear_memory()

    model = YOLO(str(MODEL_PATH))

    print("\nBest YOLOv8m model loaded successfully.")

    # --------------------------------------------------------
    # LOAD VALIDATION DATA
    # --------------------------------------------------------

    print("\nLoading validation set...")

    validation_information = load_split_information(
        "valid"
    )

    # --------------------------------------------------------
    # VALIDATION PREDICTIONS
    # --------------------------------------------------------

    validation_predictions = generate_predictions(
        model=model,
        dataset_information=validation_information,
        split_name="validation",
    )

    # --------------------------------------------------------
    # TEST ALL THRESHOLDS ON VALIDATION SET
    # --------------------------------------------------------

    validation_results = []

    for threshold in CONFIDENCE_THRESHOLDS:
        result = evaluate_at_threshold(
            dataset_information=validation_information,
            saved_predictions=validation_predictions,
            confidence_threshold=threshold,
        )

        validation_results.append(result)

    print_threshold_table(
        split_name="validation",
        results=validation_results,
    )

    # Highest validation F1 threshold নির্বাচন
    best_validation_result = max(
        validation_results,
        key=lambda item: item["F1-score"],
    )

    best_threshold = best_validation_result[
        "Confidence"
    ]

    print("\n" + "=" * 80)
    print("SELECTED THRESHOLD FROM VALIDATION SET")
    print("=" * 80)

    print(
        f"Selected confidence : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation precision: "
        f"{best_validation_result['Precision']:.6f}"
    )

    print(
        f"Validation recall   : "
        f"{best_validation_result['Recall']:.6f}"
    )

    print(
        f"Validation F1-score : "
        f"{best_validation_result['F1-score']:.6f}"
    )

    save_validation_results(
        validation_results
    )

    # Validation predictions আর প্রয়োজন নেই
    del validation_predictions
    del validation_information

    clear_memory()

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    print("\nLoading independent test set...")

    test_information = load_split_information(
        "test"
    )

    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    test_predictions = generate_predictions(
        model=model,
        dataset_information=test_information,
        split_name="test",
    )

    # Validation-selected threshold test set-এ প্রয়োগ
    test_result = evaluate_at_threshold(
        dataset_information=test_information,
        saved_predictions=test_predictions,
        confidence_threshold=best_threshold,
    )

    # --------------------------------------------------------
    # STANDARD mAP
    # --------------------------------------------------------

    map_results = evaluate_standard_map(
        model=model
    )

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL YOLOv8m TEST RESULTS")
    print("=" * 80)

    print(
        f"Selected confidence : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Evaluation IoU      : "
        f"{EVALUATION_IOU_THRESHOLD:.2f}"
    )

    print(
        f"Test images         : "
        f"{len(test_information)}"
    )

    print(f"TP                  : {test_result['TP']}")
    print(f"FP                  : {test_result['FP']}")
    print(f"FN                  : {test_result['FN']}")

    print(
        f"Precision           : "
        f"{test_result['Precision']:.6f}"
    )

    print(
        f"Recall              : "
        f"{test_result['Recall']:.6f}"
    )

    print(
        f"F1-score            : "
        f"{test_result['F1-score']:.6f}"
    )

    print(
        f"mAP@0.5             : "
        f"{map_results['mAP@0.5']:.6f}"
    )

    print(
        f"mAP@0.5:0.95        : "
        f"{map_results['mAP@0.5:0.95']:.6f}"
    )

    print(
        f"Ultralytics P       : "
        f"{map_results['Ultralytics Precision']:.6f}"
    )

    print(
        f"Ultralytics R       : "
        f"{map_results['Ultralytics Recall']:.6f}"
    )

    print("\nPer-class final test results")
    print("-" * 100)

    for class_result in test_result[
        "Class results"
    ]:
        print(
            f"{class_result['Class']:<15}"
            f"TP={class_result['TP']:<5}"
            f"FP={class_result['FP']:<5}"
            f"FN={class_result['FN']:<5}"
            f"P={class_result['Precision']:.4f}  "
            f"R={class_result['Recall']:.4f}  "
            f"F1={class_result['F1-score']:.4f}"
        )

    # --------------------------------------------------------
    # SAVE FINAL RESULTS
    # --------------------------------------------------------

    save_final_test_results(
        best_threshold=best_threshold,
        test_result=test_result,
        map_results=map_results,
    )

    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()