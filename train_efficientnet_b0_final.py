# ============================================================
# EFFICIENTNET-B0 FABRIC DEFECT CLASSIFICATION
# PART 1: SETUP, DATASET, MODEL AND OPTIMIZER
# ============================================================

from pathlib import Path
from collections import Counter
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "classification_dataset"

TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"

RESULT_DIR = (
    ROOT_DIR
    / "results"
    / "efficientnet_b0_classification"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_SAVE_PATH = (
    RESULT_DIR
    / "best_efficientnet_b0.pt"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASSES = [
    "Cut",
    "Hole",
    "Stain",
    "ThreadError"
]

NUM_CLASSES = len(CLASSES)


# ============================================================
# TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 100

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.0001

PATIENCE = 15

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("EFFICIENTNET-B0 FABRIC DEFECT CLASSIFICATION")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# PRETRAINED WEIGHTS
# ============================================================

weights = (
    EfficientNet_B0_Weights.IMAGENET1K_V1
)

imagenet_mean = weights.transforms().mean

imagenet_std = weights.transforms().std


# ============================================================
# DATA TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (256, 256)
    ),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.80, 1.00),
        ratio=(0.90, 1.10)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.20
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )

])


valid_test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )

])


# ============================================================
# LOAD DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(

    root=TRAIN_DIR,

    transform=train_transform

)


valid_dataset = datasets.ImageFolder(

    root=VALID_DIR,

    transform=valid_test_transform

)


test_dataset = datasets.ImageFolder(

    root=TEST_DIR,

    transform=valid_test_transform

)


# ============================================================
# CHECK CLASS MAPPING
# ============================================================

expected_mapping = {

    class_name: class_index

    for class_index, class_name in enumerate(
        CLASSES
    )

}


print("\nClass Mapping:")

print(
    train_dataset.class_to_idx
)


if train_dataset.class_to_idx != expected_mapping:

    raise ValueError(

        "\nClass mapping mismatch.\n"

        f"Found: "
        f"{train_dataset.class_to_idx}\n"

        f"Expected: "
        f"{expected_mapping}\n"

        "Check the class folder names."

    )


if valid_dataset.class_to_idx != expected_mapping:

    raise ValueError(

        "Validation class mapping does not "
        "match training."

    )


if test_dataset.class_to_idx != expected_mapping:

    raise ValueError(

        "Test class mapping does not "
        "match training."

    )


# ============================================================
# DATASET DISTRIBUTION
# ============================================================

train_class_counts = Counter(

    train_dataset.targets

)


print("\nTraining Distribution")

for class_index, class_name in enumerate(
    CLASSES
):

    number_of_images = (
        train_class_counts[class_index]
    )

    print(

        f"{class_name:<15}: "
        f"{number_of_images}"

    )


print("\nDataset Ready")

print(
    f"Train: {len(train_dataset)}"
)

print(
    f"Valid: {len(valid_dataset)}"
)

print(
    f"Test : {len(test_dataset)}"
)


# ============================================================
# WEIGHTED RANDOM SAMPLER
# ============================================================

total_train_samples = len(
    train_dataset
)


class_weights = {}


for class_index in range(
    NUM_CLASSES
):

    class_count = (
        train_class_counts[class_index]
    )

    if class_count == 0:

        raise ValueError(

            f"No training samples found for "
            f"class: {CLASSES[class_index]}"

        )

    class_weights[class_index] = (

        total_train_samples

        /

        (
            NUM_CLASSES
            * class_count
        )

    )


sample_weights = [

    class_weights[label]

    for label in train_dataset.targets

]


train_sampler = WeightedRandomSampler(

    weights=sample_weights,

    num_samples=len(sample_weights),

    replacement=True

)


print("\nClass Sampling Weights")

for class_index, class_name in enumerate(
    CLASSES
):

    print(

        f"{class_name:<15}: "
        f"{class_weights[class_index]:.6f}"

    )


# ============================================================
# DATA LOADERS
# ============================================================

PIN_MEMORY = (
    DEVICE.type == "cuda"
)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    sampler=train_sampler,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY

)


valid_loader = DataLoader(

    valid_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY

)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY

)


print("\nDataLoaders Created")

print(
    f"Training batches  : "
    f"{len(train_loader)}"
)

print(
    f"Validation batches: "
    f"{len(valid_loader)}"
)

print(
    f"Testing batches   : "
    f"{len(test_loader)}"
)


# ============================================================
# LOAD PRETRAINED EFFICIENTNET-B0
# ============================================================

model = efficientnet_b0(

    weights=weights

)


# Replace the final classification layer

input_features = (
    model.classifier[1].in_features
)


model.classifier[1] = nn.Linear(

    input_features,

    NUM_CLASSES

)


model = model.to(
    DEVICE
)


print("\nModel Loaded Successfully")

print(
    "Model: EfficientNet-B0"
)

print(
    f"Number of classes: "
    f"{NUM_CLASSES}"
)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=5,

        min_lr=1e-7

    )
)


# ============================================================
# MIXED PRECISION SCALER
# ============================================================

scaler = torch.amp.GradScaler(

    device=DEVICE.type,

    enabled=DEVICE.type == "cuda"

)


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {

    "epoch": [],

    "train_loss": [],

    "valid_loss": [],

    "train_accuracy": [],

    "valid_accuracy": [],

    "valid_f1": [],

    "learning_rate": []

}


best_valid_f1 = 0.0

patience_counter = 0


print("\nPart 1 completed successfully.")

print(
    "Dataset, EfficientNet-B0 model, "
    "optimizer and scheduler are ready."
)
# ============================================================
# EFFICIENTNET-B0 FABRIC DEFECT CLASSIFICATION
# PART 2: TRAINING, VALIDATION, TESTING AND VISUALIZATION
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# ============================================================
# TRAINING AND VALIDATION LOOP
# ============================================================

for epoch in range(EPOCHS):

    print("\n")
    print("=" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 60)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    running_train_loss = 0.0

    train_true = []
    train_pred = []

    for images, labels in tqdm(
        train_loader,
        desc="Training"
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=DEVICE.type == "cuda"
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        if DEVICE.type == "cuda":

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)

            scaler.update()

        else:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

        running_train_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_true.extend(
            labels.detach().cpu().numpy()
        )

        train_pred.extend(
            predictions.detach().cpu().numpy()
        )

    train_loss = (
        running_train_loss
        /
        len(train_dataset)
    )

    train_accuracy = accuracy_score(
        train_true,
        train_pred
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    running_valid_loss = 0.0

    valid_true = []
    valid_pred = []

    with torch.no_grad():

        for images, labels in tqdm(
            valid_loader,
            desc="Validation"
        ):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=DEVICE.type == "cuda"
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            running_valid_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            valid_true.extend(
                labels.detach().cpu().numpy()
            )

            valid_pred.extend(
                predictions.detach().cpu().numpy()
            )

    valid_loss = (
        running_valid_loss
        /
        len(valid_dataset)
    )

    valid_accuracy = accuracy_score(
        valid_true,
        valid_pred
    )

    valid_f1 = f1_score(
        valid_true,
        valid_pred,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0
    )

    scheduler.step(
        valid_f1
    )

    current_learning_rate = (
        optimizer.param_groups[0]["lr"]
    )

    # --------------------------------------------------------
    # STORE HISTORY
    # --------------------------------------------------------

    history["epoch"].append(
        epoch + 1
    )

    history["train_loss"].append(
        train_loss
    )

    history["valid_loss"].append(
        valid_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["valid_accuracy"].append(
        valid_accuracy
    )

    history["valid_f1"].append(
        valid_f1
    )

    history["learning_rate"].append(
        current_learning_rate
    )

    # --------------------------------------------------------
    # PRINT EPOCH RESULT
    # --------------------------------------------------------

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Valid Loss : {valid_loss:.4f}"
    )

    print(
        f"Train Acc  : {train_accuracy:.4f}"
    )

    print(
        f"Valid Acc  : {valid_accuracy:.4f}"
    )

    print(
        f"Valid F1   : {valid_f1:.4f}"
    )

    print(
        f"Learning Rate: "
        f"{current_learning_rate:.8f}"
    )

    # --------------------------------------------------------
    # SAVE HISTORY AFTER EVERY EPOCH
    # --------------------------------------------------------

    history_dataframe = pd.DataFrame(
        history
    )

    history_dataframe.to_csv(
        RESULT_DIR / "training_history.csv",
        index=False
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if valid_f1 > best_valid_f1:

        best_valid_f1 = valid_f1

        patience_counter = 0

        checkpoint = {

            "epoch":
                epoch + 1,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "best_valid_f1":
                best_valid_f1,

            "class_names":
                CLASSES,

            "class_to_idx":
                train_dataset.class_to_idx

        }

        torch.save(
            checkpoint,
            MODEL_SAVE_PATH
        )

        print(
            "Best model saved successfully."
        )

    else:

        patience_counter += 1

        print(
            f"No improvement: "
            f"{patience_counter}/{PATIENCE}"
        )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping activated."
        )

        break


print("\nTraining Completed")


# ============================================================
# SAVE FINAL TRAINING HISTORY
# ============================================================

history_dataframe = pd.DataFrame(
    history
)

history_dataframe.to_csv(
    RESULT_DIR / "training_history.csv",
    index=False
)


# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history["epoch"],
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    history["epoch"],
    history["valid_loss"],
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "EfficientNet-B0 Training and Validation Loss"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# ACCURACY CURVE
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history["epoch"],
    history["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history["epoch"],
    history["valid_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "EfficientNet-B0 Training and Validation Accuracy"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "accuracy_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# VALIDATION F1 CURVE
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history["epoch"],
    history["valid_f1"],
    label="Validation F1-score"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "F1-score"
)

plt.title(
    "EfficientNet-B0 Validation F1-score"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "validation_f1_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# LEARNING RATE CURVE
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history["epoch"],
    history["learning_rate"],
    label="Learning Rate"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Learning Rate"
)

plt.title(
    "EfficientNet-B0 Learning Rate Schedule"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "learning_rate_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# LOAD BEST MODEL
# ============================================================

print(
    "\nLoading Best EfficientNet-B0 Model"
)

checkpoint = torch.load(
    MODEL_SAVE_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(
    DEVICE
)

print(
    f"Best Epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Best Validation F1: "
    f"{checkpoint['best_valid_f1']:.6f}"
)


# ============================================================
# TESTING
# ============================================================

print(
    "\nTesting Best Model"
)

model.eval()

running_test_loss = 0.0

test_true = []
test_pred = []
test_probabilities = []

with torch.no_grad():

    for images, labels in tqdm(
        test_loader,
        desc="Testing"
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=DEVICE.type == "cuda"
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        running_test_loss += (
            loss.item() * images.size(0)
        )

        test_true.extend(
            labels.detach().cpu().numpy()
        )

        test_pred.extend(
            predictions.detach().cpu().numpy()
        )

        test_probabilities.extend(
            probabilities.detach().cpu().numpy()
        )


test_loss = (
    running_test_loss
    /
    len(test_dataset)
)


# ============================================================
# FINAL METRICS
# ============================================================

test_accuracy = accuracy_score(
    test_true,
    test_pred
)

test_precision = precision_score(
    test_true,
    test_pred,
    labels=list(range(NUM_CLASSES)),
    average="macro",
    zero_division=0
)

test_recall = recall_score(
    test_true,
    test_pred,
    labels=list(range(NUM_CLASSES)),
    average="macro",
    zero_division=0
)

test_f1 = f1_score(
    test_true,
    test_pred,
    labels=list(range(NUM_CLASSES)),
    average="macro",
    zero_division=0
)

weighted_precision = precision_score(
    test_true,
    test_pred,
    labels=list(range(NUM_CLASSES)),
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    test_true,
    test_pred,
    labels=list(range(NUM_CLASSES)),
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    test_true,
    test_pred,
    labels=list(range(NUM_CLASSES)),
    average="weighted",
    zero_division=0
)


print("\n")
print("=" * 70)
print("FINAL EFFICIENTNET-B0 TEST RESULTS")
print("=" * 70)

print(
    f"Test Loss          : "
    f"{test_loss:.6f}"
)

print(
    f"Accuracy           : "
    f"{test_accuracy:.6f}"
)

print(
    f"Precision          : "
    f"{test_precision:.6f}"
)

print(
    f"Recall             : "
    f"{test_recall:.6f}"
)

print(
    f"F1-score           : "
    f"{test_f1:.6f}"
)

print(
    f"Weighted Precision : "
    f"{weighted_precision:.6f}"
)

print(
    f"Weighted Recall    : "
    f"{weighted_recall:.6f}"
)

print(
    f"Weighted F1-score  : "
    f"{weighted_f1:.6f}"
)


# ============================================================
# SAVE TEST RESULTS
# ============================================================

with open(
    RESULT_DIR / "test_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "EFFICIENTNET-B0 FABRIC DEFECT CLASSIFICATION\n"
    )

    file.write(
        "=" * 55 + "\n"
    )

    file.write(
        f"Best Epoch: "
        f"{checkpoint['epoch']}\n"
    )

    file.write(
        f"Best Validation F1: "
        f"{checkpoint['best_valid_f1']:.6f}\n"
    )

    file.write(
        f"Test Loss: "
        f"{test_loss:.6f}\n"
    )

    file.write(
        f"Accuracy: "
        f"{test_accuracy:.6f}\n"
    )

    file.write(
        f"Precision: "
        f"{test_precision:.6f}\n"
    )

    file.write(
        f"Recall: "
        f"{test_recall:.6f}\n"
    )

    file.write(
        f"F1-score: "
        f"{test_f1:.6f}\n"
    )

    file.write(
        f"Weighted Precision: "
        f"{weighted_precision:.6f}\n"
    )

    file.write(
        f"Weighted Recall: "
        f"{weighted_recall:.6f}\n"
    )

    file.write(
        f"Weighted F1-score: "
        f"{weighted_f1:.6f}\n"
    )


# ============================================================
# TEST SUMMARY CSV
# ============================================================

test_summary = pd.DataFrame([{

    "Model":
        "EfficientNet-B0",

    "Best_Epoch":
        checkpoint["epoch"],

    "Best_Validation_F1":
        checkpoint["best_valid_f1"],

    "Test_Loss":
        test_loss,

    "Accuracy":
        test_accuracy,

    "Precision":
        test_precision,

    "Recall":
        test_recall,

    "F1_score":
        test_f1,

    "Weighted_Precision":
        weighted_precision,

    "Weighted_Recall":
        weighted_recall,

    "Weighted_F1_score":
        weighted_f1

}])

test_summary.to_csv(
    RESULT_DIR / "test_summary.csv",
    index=False
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report_dictionary = classification_report(
    test_true,
    test_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASSES,
    output_dict=True,
    zero_division=0
)

report_dataframe = pd.DataFrame(
    report_dictionary
).transpose()

report_dataframe.to_csv(
    RESULT_DIR / "classification_report.csv"
)


print("\nPer-Class Classification Results")
print("-" * 70)

for class_name in CLASSES:

    class_result = report_dictionary[
        class_name
    ]

    print(
        f"{class_name:<15} "
        f"Precision={class_result['precision']:.4f} "
        f"Recall={class_result['recall']:.4f} "
        f"F1={class_result['f1-score']:.4f} "
        f"Support={int(class_result['support'])}"
    )


# ============================================================
# SAVE INDIVIDUAL TEST PREDICTIONS
# ============================================================

prediction_rows = []

for sample_index, (
    true_label,
    predicted_label,
    probabilities
) in enumerate(
    zip(
        test_true,
        test_pred,
        test_probabilities
    )
):

    image_path = (
        test_dataset.samples[
            sample_index
        ][0]
    )

    predicted_confidence = float(
        np.max(probabilities)
    )

    row = {

        "image_path":
            image_path,

        "true_class":
            CLASSES[true_label],

        "predicted_class":
            CLASSES[predicted_label],

        "correct":
            int(
                true_label == predicted_label
            ),

        "confidence":
            predicted_confidence

    }

    for class_index, class_name in enumerate(
        CLASSES
    ):

        row[
            f"probability_{class_name}"
        ] = float(
            probabilities[class_index]
        )

    prediction_rows.append(
        row
    )


prediction_dataframe = pd.DataFrame(
    prediction_rows
)

prediction_dataframe.to_csv(
    RESULT_DIR / "test_predictions.csv",
    index=False
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion_matrix_values = confusion_matrix(
    test_true,
    test_pred,
    labels=list(range(NUM_CLASSES))
)

plt.figure(
    figsize=(8, 7)
)

sns.heatmap(
    confusion_matrix_values,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASSES,
    yticklabels=CLASSES
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.title(
    "EfficientNet-B0 Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================

confusion_matrix_normalized = (
    confusion_matrix_values.astype(
        np.float64
    )
)

row_sums = confusion_matrix_normalized.sum(
    axis=1,
    keepdims=True
)

confusion_matrix_normalized = np.divide(
    confusion_matrix_normalized,
    row_sums,
    out=np.zeros_like(
        confusion_matrix_normalized
    ),
    where=row_sums != 0
)

plt.figure(
    figsize=(8, 7)
)

sns.heatmap(
    confusion_matrix_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=CLASSES,
    yticklabels=CLASSES
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.title(
    "EfficientNet-B0 Normalized Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "normalized_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\nAll files saved successfully at:")

print(
    RESULT_DIR
)