from pathlib import Path
import csv

import torch
from ultralytics import YOLO


# ============================================================
# PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_YAML = ROOT / "yolo_dataset" / "data.yaml"
RESULTS_DIR = ROOT / "results"

RUN_NAME = "yolov8m_fabric_960"
TEST_RUN_NAME = "yolov8m_fabric_test"

MODEL_NAME = "yolov8m.pt"

EPOCHS = 150
IMAGE_SIZE = 960
BATCH_SIZE = 8
WORKERS = 4
PATIENCE = 30
SEED = 42


# ============================================================
# SAVE FINAL METRICS
# ============================================================

def save_test_metrics(metrics, model, output_folder):
    output_folder.mkdir(parents=True, exist_ok=True)

    summary_path = output_folder / "test_metrics.txt"
    csv_path = output_folder / "test_metrics.csv"

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("YOLOv8m Fabric Defect Detection\n")
        file.write("=" * 40 + "\n")
        file.write(f"Precision: {precision:.6f}\n")
        file.write(f"Recall: {recall:.6f}\n")
        file.write(f"mAP@0.5: {map50:.6f}\n")
        file.write(f"mAP@0.5:0.95: {map5095:.6f}\n")

        file.write("\nPer-class mAP@0.5\n")
        file.write("-" * 40 + "\n")

        for class_id, class_ap50 in enumerate(metrics.box.ap50):
            class_name = model.names[class_id]

            file.write(
                f"{class_name}: {float(class_ap50):.6f}\n"
            )

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "Class",
                "Precision",
                "Recall",
                "mAP50",
                "mAP50-95",
            ]
        )

        writer.writerow(
            [
                "Overall",
                precision,
                recall,
                map50,
                map5095,
            ]
        )

        for class_id, class_ap50 in enumerate(metrics.box.ap50):
            writer.writerow(
                [
                    model.names[class_id],
                    "",
                    "",
                    float(class_ap50),
                    "",
                ]
            )

    print(f"Metrics text saved: {summary_path}")
    print(f"Metrics CSV saved : {csv_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("YOLOv8m Fabric Defect Detection")
    print("=" * 60)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml পাওয়া যায়নি:\n{DATA_YAML}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU পাওয়া যাচ্ছে না।"
        )

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print(f"Dataset          : {DATA_YAML}")
    print(f"Model            : {MODEL_NAME}")

    # --------------------------------------------------------
    # LOAD PRETRAINED MODEL
    # --------------------------------------------------------

    model = YOLO(MODEL_NAME)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    training_results = model.train(
        data=str(DATA_YAML),

        epochs=EPOCHS,
        patience=PATIENCE,

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=0,
        workers=WORKERS,

        project=str(RESULTS_DIR),
        name=RUN_NAME,
        exist_ok=True,

        pretrained=True,

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,

        # Augmentation
        degrees=5.0,
        translate=0.10,
        scale=0.40,
        shear=2.0,
        perspective=0.0002,

        fliplr=0.5,
        flipud=0.2,

        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,

        mosaic=0.7,
        mixup=0.1,
        close_mosaic=15,

        # Saving and visualization
        save=True,
        save_period=-1,
        plots=True,

        cache="disk",
        amp=True,
        verbose=True,

        seed=SEED,
        deterministic=True,
    )

    print("\nTraining completed.")
    print(f"Training folder: {training_results.save_dir}")

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    best_model_path = (
        RESULTS_DIR
        / RUN_NAME
        / "weights"
        / "best.pt"
    )

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"best.pt পাওয়া যায়নি:\n{best_model_path}"
        )

    print(f"Best model: {best_model_path}")

    best_model = YOLO(str(best_model_path))

    # --------------------------------------------------------
    # TEST EVALUATION
    # --------------------------------------------------------

    test_metrics = best_model.val(
        data=str(DATA_YAML),

        split="test",

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=0,
        workers=WORKERS,

        project=str(RESULTS_DIR),
        name=TEST_RUN_NAME,
        exist_ok=True,

        plots=True,

        save_json=False,
        verbose=True,
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Precision     : {test_metrics.box.mp:.6f}")
    print(f"Recall        : {test_metrics.box.mr:.6f}")
    print(f"mAP@0.5       : {test_metrics.box.map50:.6f}")
    print(f"mAP@0.5:0.95  : {test_metrics.box.map:.6f}")

    print("\nPer-class mAP@0.5")

    for class_id, ap50 in enumerate(test_metrics.box.ap50):
        class_name = best_model.names[class_id]

        print(
            f"{class_id}: {class_name:<15} "
            f"{float(ap50):.6f}"
        )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    test_output_folder = RESULTS_DIR / TEST_RUN_NAME

    save_test_metrics(
        metrics=test_metrics,
        model=best_model,
        output_folder=test_output_folder,
    )

    print("\nGenerated training graphs:")

    training_folder = RESULTS_DIR / RUN_NAME

    graph_files = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "labels.jpg",
        "labels_correlogram.jpg",
    ]

    for filename in graph_files:
        file_path = training_folder / filename

        if file_path.exists():
            print(f"[Created] {file_path}")
        else:
            print(f"[Not found] {file_path}")

    print("\nGenerated test graphs:")

    test_graph_files = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
    ]

    for filename in test_graph_files:
        file_path = test_output_folder / filename

        if file_path.exists():
            print(f"[Created] {file_path}")
        else:
            print(f"[Not found] {file_path}")

    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    main()