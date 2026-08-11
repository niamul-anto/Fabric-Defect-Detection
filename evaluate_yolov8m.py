from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision.ops import box_iou
from tqdm import tqdm
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATASET_ROOT = ROOT / "yolo_dataset"
TEST_IMAGE_DIR = DATASET_ROOT / "test" / "images"
TEST_LABEL_DIR = DATASET_ROOT / "test" / "labels"
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
    / "corrected_evaluation"
)

CLASS_NAMES = [
    "Cut",
    "Hole",
    "Stain",
    "ThreadError",
]

NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 960
BATCH_SIZE = 8
WORKERS = 4

# Same thresholds used for Faster R-CNN and RetinaNet
SCORE_THRESHOLD = 0.50
EVALUATION_IOU_THRESHOLD = 0.50

# YOLO NMS threshold, not evaluation IoU
NMS_IOU_THRESHOLD = 0.70

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# READ YOLO GROUND-TRUTH LABELS
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

    text = label_path.read_text(encoding="utf-8").strip()

    if not text:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )

    for line in text.splitlines():
        if not line.strip():
            continue

        values = line.split()

        if len(values) != 5:
            print(
                f"Warning: invalid annotation ignored: "
                f"{label_path.name}: {line}"
            )
            continue

        try:
            class_id, x_center, y_center, width, height = map(
                float,
                values,
            )
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

        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height

        xmin = x_center - width / 2
        ymin = y_center - height / 2
        xmax = x_center + width / 2
        ymax = y_center + height / 2

        xmin = max(0.0, min(xmin, image_width - 1.0))
        ymin = max(0.0, min(ymin, image_height - 1.0))
        xmax = max(1.0, min(xmax, float(image_width)))
        ymax = max(1.0, min(ymax, float(image_height)))

        if xmax <= xmin or ymax <= ymin:
            print(
                f"Warning: invalid box ignored: "
                f"{label_path.name}: {line}"
            )
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_id)

    if boxes:
        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64),
        )

    return (
        torch.zeros((0, 4), dtype=torch.float32),
        torch.zeros((0,), dtype=torch.int64),
    )


# ============================================================
# TRUE PRECISION, RECALL AND F1
# ============================================================

@torch.inference_mode()
def evaluate_true_metrics(model: YOLO):
    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    if not TEST_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Test image folder not found:\n{TEST_IMAGE_DIR}"
        )

    image_paths = sorted(
        image_path
        for image_path in TEST_IMAGE_DIR.iterdir()
        if image_path.suffix.lower() in supported_extensions
    )

    if not image_paths:
        raise RuntimeError(
            f"No test images found in:\n{TEST_IMAGE_DIR}"
        )

    class_statistics = {
        class_id: {
            "TP": 0,
            "FP": 0,
            "FN": 0,
        }
        for class_id in range(NUM_CLASSES)
    }

    for image_path in tqdm(
        image_paths,
        desc="True Precision Recall F1",
    ):
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        label_path = TEST_LABEL_DIR / f"{image_path.stem}.txt"

        gt_boxes, gt_labels = read_yolo_labels(
            label_path=label_path,
            image_width=image_width,
            image_height=image_height,
        )

        result = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=SCORE_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            device=DEVICE,
            verbose=False,
            max_det=300,
        )[0]

        if result.boxes is not None and len(result.boxes) > 0:
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

            sorted_indices = torch.argsort(
                pred_scores,
                descending=True,
            )

            pred_boxes = pred_boxes[sorted_indices]
            pred_labels = pred_labels[sorted_indices]

        else:
            pred_boxes = torch.zeros(
                (0, 4),
                dtype=torch.float32,
            )

            pred_labels = torch.zeros(
                (0,),
                dtype=torch.int64,
            )

        for class_id in range(NUM_CLASSES):
            class_gt_boxes = gt_boxes[
                gt_labels == class_id
            ]

            class_pred_boxes = pred_boxes[
                pred_labels == class_id
            ]

            matched_ground_truth = torch.zeros(
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

                # One ground-truth box can be matched only once
                ious[matched_ground_truth] = -1.0

                best_iou, best_gt_index = torch.max(
                    ious,
                    dim=0,
                )

                if float(best_iou) >= EVALUATION_IOU_THRESHOLD:
                    class_tp += 1
                    matched_ground_truth[best_gt_index] = True
                else:
                    class_fp += 1

            class_fn = int(
                (~matched_ground_truth).sum().item()
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

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)

    f1_score = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

    per_class_results = {}

    for class_id in range(NUM_CLASSES):
        tp = class_statistics[class_id]["TP"]
        fp = class_statistics[class_id]["FP"]
        fn = class_statistics[class_id]["FN"]

        class_precision = tp / max(tp + fp, 1)
        class_recall = tp / max(tp + fn, 1)

        class_f1 = (
            2 * class_precision * class_recall
            / max(class_precision + class_recall, 1e-12)
        )

        per_class_results[CLASS_NAMES[class_id]] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": class_precision,
            "Recall": class_recall,
            "F1-score": class_f1,
        }

    return {
        "Number of test images": len(image_paths),
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_score,
        "per_class": per_class_results,
    }


# ============================================================
# ULTRALYTICS TEST mAP
# ============================================================

def evaluate_map(model: YOLO):
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,

        # Low confidence is used to build full PR curves for AP
        conf=0.001,

        # NMS IoU threshold
        iou=NMS_IOU_THRESHOLD,

        max_det=300,
        plots=False,
        save_json=False,
        verbose=False,
    )

    return {
        "mAP@0.5": float(metrics.box.map50),
        "mAP@0.5:0.95": float(metrics.box.map),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nYOLOv8m best.pt not found:\n{MODEL_PATH}\n\n"
            "Check whether the model exists inside:\n"
            "results\\yolov8m_fabric_960\\weights\\best.pt"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml not found:\n{DATA_YAML}"
        )

    print("=" * 70)
    print("YOLOv8m Corrected Test Evaluation")
    print("=" * 70)

    print(f"Device      : {DEVICE}")
    print(f"Model path  : {MODEL_PATH}")
    print(f"Test folder : {TEST_IMAGE_DIR}")

    if torch.cuda.is_available():
        print(
            f"GPU         : "
            f"{torch.cuda.get_device_name(0)}"
        )

    model = YOLO(str(MODEL_PATH))

    print("\nBest YOLOv8m model loaded successfully.")

    true_metrics = evaluate_true_metrics(model)
    map_metrics = evaluate_map(model)

    print("\n" + "=" * 70)
    print("FINAL CORRECTED YOLOv8m TEST RESULTS")
    print("=" * 70)

    print(
        f"Test images          : "
        f"{true_metrics['Number of test images']}"
    )
    print(
        f"Confidence threshold : "
        f"{SCORE_THRESHOLD:.2f}"
    )
    print(
        f"IoU threshold        : "
        f"{EVALUATION_IOU_THRESHOLD:.2f}"
    )
    print(f"TP                   : {true_metrics['TP']}")
    print(f"FP                   : {true_metrics['FP']}")
    print(f"FN                   : {true_metrics['FN']}")
    print(
        f"True Precision       : "
        f"{true_metrics['Precision']:.6f}"
    )
    print(
        f"True Recall          : "
        f"{true_metrics['Recall']:.6f}"
    )
    print(
        f"True F1-score        : "
        f"{true_metrics['F1-score']:.6f}"
    )
    print(
        f"mAP@0.5              : "
        f"{map_metrics['mAP@0.5']:.6f}"
    )
    print(
        f"mAP@0.5:0.95         : "
        f"{map_metrics['mAP@0.5:0.95']:.6f}"
    )

    print("\nPer-class true results")
    print("-" * 90)

    for class_name, values in true_metrics["per_class"].items():
        print(
            f"{class_name:<15} "
            f"TP={values['TP']:<4} "
            f"FP={values['FP']:<4} "
            f"FN={values['FN']:<4} "
            f"P={values['Precision']:.4f} "
            f"R={values['Recall']:.4f} "
            f"F1={values['F1-score']:.4f}"
        )

    overall_row = {
        "Model": "YOLOv8m",
        "Confidence threshold": SCORE_THRESHOLD,
        "IoU threshold": EVALUATION_IOU_THRESHOLD,
        "Test images": true_metrics["Number of test images"],
        "TP": true_metrics["TP"],
        "FP": true_metrics["FP"],
        "FN": true_metrics["FN"],
        "Precision": true_metrics["Precision"],
        "Recall": true_metrics["Recall"],
        "F1-score": true_metrics["F1-score"],
        "mAP@0.5": map_metrics["mAP@0.5"],
        "mAP@0.5:0.95": map_metrics["mAP@0.5:0.95"],
    }

    overall_csv_path = (
        OUTPUT_DIR / "yolov8m_true_test_results.csv"
    )

    pd.DataFrame(
        [overall_row]
    ).to_csv(
        overall_csv_path,
        index=False,
    )

    class_rows = []

    for class_name, values in true_metrics["per_class"].items():
        class_rows.append({
            "Class": class_name,
            "TP": values["TP"],
            "FP": values["FP"],
            "FN": values["FN"],
            "Precision": values["Precision"],
            "Recall": values["Recall"],
            "F1-score": values["F1-score"],
        })

    class_csv_path = (
        OUTPUT_DIR / "yolov8m_true_class_results.csv"
    )

    pd.DataFrame(
        class_rows
    ).to_csv(
        class_csv_path,
        index=False,
    )

    print("\nSaved files:")
    print(overall_csv_path)
    print(class_csv_path)


if __name__ == "__main__":
    main()