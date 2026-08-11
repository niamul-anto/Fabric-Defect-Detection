from pathlib import Path

import torch
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_YAML = PROJECT_ROOT / "yolo_dataset" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "results"

MODEL_NAME = "yolov8n.pt"

EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16
PATIENCE = 15
WORKERS = 4
SEED = 42


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 60)
    print("YOLOv8 Fabric Defect Detection Training")
    print("=" * 60)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml পাওয়া যায়নি:\n{DATA_YAML}"
        )

    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU পাওয়া যায়নি। CPU ব্যবহার করা হবে।")

    print(f"Dataset YAML     : {DATA_YAML}")
    print(f"Model            : {MODEL_NAME}")
    print(f"Device           : {device}")

    # Pretrained YOLOv8 Nano model
    model = YOLO(MODEL_NAME)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    train_results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        workers=WORKERS,
        patience=PATIENCE,
        seed=SEED,

        project=str(OUTPUT_DIR),
        name="yolov8n_fabric",
        exist_ok=True,

        pretrained=True,
        optimizer="auto",
        verbose=True,

        save=True,
        save_period=-1,
        plots=True,

        cache=False,
        deterministic=True,
    )

    print("\nTraining completed.")
    print(f"Training results: {train_results.save_dir}")

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    best_model_path = (
        OUTPUT_DIR
        / "yolov8n_fabric"
        / "weights"
        / "best.pt"
    )

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Best model পাওয়া যায়নি:\n{best_model_path}"
        )

    print(f"\nBest model: {best_model_path}")

    best_model = YOLO(str(best_model_path))

    # --------------------------------------------------------
    # TEST SET EVALUATION
    # --------------------------------------------------------

    test_results = best_model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        workers=WORKERS,

        project=str(OUTPUT_DIR),
        name="yolov8n_test",
        exist_ok=True,

        plots=True,
        save_json=False,
    )

    print("\nTest evaluation completed.")

    print(f"Precision      : {test_results.box.mp:.6f}")
    print(f"Recall         : {test_results.box.mr:.6f}")
    print(f"mAP@0.5        : {test_results.box.map50:.6f}")
    print(f"mAP@0.5:0.95   : {test_results.box.map:.6f}")

    print("\nPer-class mAP@0.5:")

    class_names = best_model.names
    per_class_map50 = test_results.box.ap50

    for class_id, class_map50 in enumerate(per_class_map50):
        class_name = class_names[class_id]

        print(
            f"{class_id}: {class_name:<15} "
            f"mAP@0.5 = {float(class_map50):.6f}"
        )

    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    main()