from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATASET_ROOT = ROOT / "yolo_dataset"

SAVE_DIR = ROOT / "results" / "retinanet_fabric"

MODEL_PATH = SAVE_DIR / "best_model.pth"

OUTPUT_DIR = SAVE_DIR / "corrected_evaluation"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "Background",
    "Cut",
    "Hole",
    "Stain",
    "ThreadError",
]

NUM_CLASSES = len(CLASS_NAMES)

BATCH_SIZE = 4
NUM_WORKERS = 0

IMAGE_MIN = 960
IMAGE_MAX = 1280

SCORE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50


# ============================================================
# DATASET
# ============================================================

class FabricDataset(Dataset):
    def __init__(self, split_dir: Path):
        self.image_dir = split_dir / "images"
        self.label_dir = split_dir / "labels"

        supported_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }

        self.images = sorted(
            image_path
            for image_path in self.image_dir.iterdir()
            if image_path.suffix.lower() in supported_extensions
        )

        if not self.images:
            raise RuntimeError(
                f"No images found in: {self.image_dir}"
            )

    def __len__(self):
        return len(self.images)

    def read_labels(
        self,
        label_path: Path,
        image_width: int,
        image_height: int,
    ):
        boxes = []
        labels = []

        if label_path.exists():
            lines = label_path.read_text(
                encoding="utf-8"
            ).strip().splitlines()

            for line in lines:
                if not line.strip():
                    continue

                values = line.split()

                if len(values) != 5:
                    print(
                        f"Warning: invalid label line ignored: "
                        f"{label_path.name}: {line}"
                    )
                    continue

                class_id, x_center, y_center, width, height = map(
                    float,
                    values,
                )

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
                        f"{label_path.name}"
                    )
                    continue

                boxes.append([xmin, ymin, xmax, ymax])

                # Class 0 is reserved for background
                labels.append(int(class_id) + 1)

        if boxes:
            boxes_tensor = torch.tensor(
                boxes,
                dtype=torch.float32,
            )

            labels_tensor = torch.tensor(
                labels,
                dtype=torch.int64,
            )
        else:
            boxes_tensor = torch.zeros(
                (0, 4),
                dtype=torch.float32,
            )

            labels_tensor = torch.zeros(
                (0,),
                dtype=torch.int64,
            )

        return boxes_tensor, labels_tensor

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        boxes, labels = self.read_labels(
            label_path,
            image_width,
            image_height,
        )

        image_tensor = TF.to_tensor(image)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(
                [index],
                dtype=torch.int64,
            ),
        }

        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================
# RETINANET MODEL
# Must match the model used during training
# ============================================================

def create_model():
    model = retinanet_resnet50_fpn_v2(
        weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
        min_size=IMAGE_MIN,
        max_size=IMAGE_MAX,
        detections_per_img=200,
    )

    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.backbone.out_channels

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=NUM_CLASSES,
        norm_layer=None,
    )

    return model


# ============================================================
# TRUE PRECISION, RECALL AND F1
# ============================================================

@torch.inference_mode()
def evaluate_true_metrics(
    model,
    loader,
    score_threshold=0.50,
    iou_threshold=0.50,
):
    model.eval()

    class_statistics = {
        class_id: {
            "TP": 0,
            "FP": 0,
            "FN": 0,
        }
        for class_id in range(1, NUM_CLASSES)
    }

    for images, targets in tqdm(
        loader,
        desc="True Precision Recall F1",
    ):
        device_images = [
            image.to(DEVICE, non_blocking=True)
            for image in images
        ]

        outputs = model(device_images)

        for output, target in zip(outputs, targets):
            gt_boxes = target["boxes"].cpu()
            gt_labels = target["labels"].cpu()

            pred_boxes = output["boxes"].detach().cpu()
            pred_labels = output["labels"].detach().cpu()
            pred_scores = output["scores"].detach().cpu()

            keep = pred_scores >= score_threshold

            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]

            sorted_indices = torch.argsort(
                pred_scores,
                descending=True,
            )

            pred_boxes = pred_boxes[sorted_indices]
            pred_labels = pred_labels[sorted_indices]

            for class_id in range(1, NUM_CLASSES):
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

                    ious[matched_ground_truth] = -1.0

                    best_iou, best_gt_index = torch.max(
                        ious,
                        dim=0,
                    )

                    if float(best_iou) >= iou_threshold:
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

    for class_id in range(1, NUM_CLASSES):
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
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_score,
        "per_class": per_class_results,
    }


# ============================================================
# COCO mAP
# ============================================================

@torch.inference_mode()
def evaluate_map(model, loader):
    model.eval()

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
    )

    for images, targets in tqdm(
        loader,
        desc="COCO mAP evaluation",
    ):
        device_images = [
            image.to(DEVICE, non_blocking=True)
            for image in images
        ]

        outputs = model(device_images)

        predictions = []
        ground_truths = []

        for output, target in zip(outputs, targets):
            predictions.append({
                "boxes": output["boxes"].detach().cpu(),
                "scores": output["scores"].detach().cpu(),
                "labels": output["labels"].detach().cpu(),
            })

            ground_truths.append({
                "boxes": target["boxes"].cpu(),
                "labels": target["labels"].cpu(),
            })

        metric.update(
            predictions,
            ground_truths,
        )

    return metric.compute()


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
            f"\nModel not found:\n{MODEL_PATH}\n\n"
            "Check the SAVE_DIR path near the top of the script."
        )

    print("=" * 70)
    print("RetinaNet Corrected Test Evaluation")
    print("=" * 70)

    print(f"Device      : {DEVICE}")
    print(f"Model path  : {MODEL_PATH}")
    print(f"Test folder : {DATASET_ROOT / 'test'}")

    if DEVICE.type == "cuda":
        print(f"GPU         : {torch.cuda.get_device_name(0)}")

    test_dataset = FabricDataset(
        DATASET_ROOT / "test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        collate_fn=collate_fn,
        drop_last=False,
    )

    print(f"Test images : {len(test_dataset)}")

    model = create_model().to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    print("\nBest RetinaNet model loaded successfully.")

    true_metrics = evaluate_true_metrics(
        model=model,
        loader=test_loader,
        score_threshold=SCORE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
    )

    map_metrics = evaluate_map(
        model=model,
        loader=test_loader,
    )

    map50 = float(map_metrics["map_50"].item())
    map5095 = float(map_metrics["map"].item())

    print("\n" + "=" * 70)
    print("FINAL CORRECTED RETINANET TEST RESULTS")
    print("=" * 70)

    print(f"Confidence threshold : {SCORE_THRESHOLD:.2f}")
    print(f"IoU threshold        : {IOU_THRESHOLD:.2f}")
    print(f"TP                   : {true_metrics['TP']}")
    print(f"FP                   : {true_metrics['FP']}")
    print(f"FN                   : {true_metrics['FN']}")
    print(f"True Precision       : {true_metrics['Precision']:.6f}")
    print(f"True Recall          : {true_metrics['Recall']:.6f}")
    print(f"True F1-score        : {true_metrics['F1-score']:.6f}")
    print(f"mAP@0.5              : {map50:.6f}")
    print(f"mAP@0.5:0.95         : {map5095:.6f}")

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
        "Model": "RetinaNet ResNet50-FPN-V2",
        "Confidence threshold": SCORE_THRESHOLD,
        "IoU threshold": IOU_THRESHOLD,
        "TP": true_metrics["TP"],
        "FP": true_metrics["FP"],
        "FN": true_metrics["FN"],
        "Precision": true_metrics["Precision"],
        "Recall": true_metrics["Recall"],
        "F1-score": true_metrics["F1-score"],
        "mAP@0.5": map50,
        "mAP@0.5:0.95": map5095,
    }

    pd.DataFrame(
        [overall_row]
    ).to_csv(
        OUTPUT_DIR / "retinanet_true_test_results.csv",
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

    pd.DataFrame(
        class_rows
    ).to_csv(
        OUTPUT_DIR / "retinanet_true_class_results.csv",
        index=False,
    )

    print("\nSaved files:")
    print(
        OUTPUT_DIR / "retinanet_true_test_results.csv"
    )
    print(
        OUTPUT_DIR / "retinanet_true_class_results.csv"
    )


if __name__ == "__main__":
    main()