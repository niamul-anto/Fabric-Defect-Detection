# ============================================================
# FD-NET: CUSTOM FABRIC DEFECT CLASSIFICATION NETWORK
# PART 1: DATASET, ARCHITECTURE AND TRAINING SETUP
#
# Architecture:
# Residual Connection
# + Depthwise Separable Convolution
# + Channel Attention
# + Spatial Attention
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


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "classification_dataset"

TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"

RESULT_DIR = ROOT_DIR / "results" / "fdnet_classification"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_SAVE_PATH = RESULT_DIR / "best_fdnet.pt"


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
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("FD-NET FABRIC DEFECT CLASSIFICATION")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# DATA TRANSFORMS
# ============================================================

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


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

    transforms.RandomVerticalFlip(
        p=0.2
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.15,
        hue=0.02
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
# VERIFY CLASS MAPPING
# ============================================================

expected_mapping = {
    class_name: class_index
    for class_index, class_name in enumerate(CLASSES)
}

print("\nClass Mapping:")
print(train_dataset.class_to_idx)

if train_dataset.class_to_idx != expected_mapping:
    raise ValueError(
        "\nClass mapping mismatch.\n"
        f"Found: {train_dataset.class_to_idx}\n"
        f"Expected: {expected_mapping}\n"
        "Check the class folder names."
    )

if valid_dataset.class_to_idx != expected_mapping:
    raise ValueError(
        "Validation class mapping does not match training."
    )

if test_dataset.class_to_idx != expected_mapping:
    raise ValueError(
        "Test class mapping does not match training."
    )


# ============================================================
# DATASET DISTRIBUTION
# ============================================================

train_class_counts = Counter(
    train_dataset.targets
)

print("\nTraining Distribution")

for class_index, class_name in enumerate(CLASSES):

    print(
        f"{class_name:<15}: "
        f"{train_class_counts[class_index]}"
    )

print("\nDataset Ready")

print(f"Train: {len(train_dataset)}")
print(f"Valid: {len(valid_dataset)}")
print(f"Test : {len(test_dataset)}")


# ============================================================
# WEIGHTED RANDOM SAMPLER
# ============================================================

total_train_samples = len(train_dataset)

class_weights = {}

for class_index in range(NUM_CLASSES):

    class_count = train_class_counts[class_index]

    if class_count == 0:
        raise ValueError(
            f"No training samples found for "
            f"class: {CLASSES[class_index]}"
        )

    class_weights[class_index] = (
        total_train_samples
        /
        (NUM_CLASSES * class_count)
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

for class_index, class_name in enumerate(CLASSES):

    print(
        f"{class_name:<15}: "
        f"{class_weights[class_index]:.6f}"
    )


# ============================================================
# DATA LOADERS
# ============================================================

PIN_MEMORY = DEVICE.type == "cuda"


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

print(f"Training batches  : {len(train_loader)}")
print(f"Validation batches: {len(valid_loader)}")
print(f"Testing batches   : {len(test_loader)}")


# ============================================================
# CONVOLUTION + BATCH NORMALIZATION + ACTIVATION
# ============================================================

class ConvBNActivation(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1
    ):

        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.SiLU(
                inplace=True
            )
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# DEPTHWISE SEPARABLE CONVOLUTION
# ============================================================

class DepthwiseSeparableConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):

        super().__init__()

        # Depthwise convolution
        self.depthwise = ConvBNActivation(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels
        )

        # Pointwise convolution
        self.pointwise = ConvBNActivation(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            groups=1
        )

    def forward(self, x):

        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


# ============================================================
# CHANNEL ATTENTION
# ============================================================

class ChannelAttention(nn.Module):

    def __init__(
        self,
        channels,
        reduction_ratio=16
    ):

        super().__init__()

        reduced_channels = max(
            channels // reduction_ratio,
            8
        )

        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.maximum_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_network = nn.Sequential(

            nn.Conv2d(
                channels,
                reduced_channels,
                kernel_size=1,
                bias=False
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                reduced_channels,
                channels,
                kernel_size=1,
                bias=False
            )
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        average_attention = self.shared_network(
            self.average_pool(x)
        )

        maximum_attention = self.shared_network(
            self.maximum_pool(x)
        )

        attention = self.sigmoid(
            average_attention + maximum_attention
        )

        return x * attention


# ============================================================
# SPATIAL ATTENTION
# ============================================================

class SpatialAttention(nn.Module):

    def __init__(
        self,
        kernel_size=7
    ):

        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        average_map = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        maximum_map, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        combined_map = torch.cat(
            [average_map, maximum_map],
            dim=1
        )

        attention = self.sigmoid(
            self.conv(combined_map)
        )

        return x * attention


# ============================================================
# COMBINED CHANNEL-SPATIAL ATTENTION
# ============================================================

class CombinedAttention(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.channel_attention = ChannelAttention(
            channels
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=7
        )

    def forward(self, x):

        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


# ============================================================
# RESIDUAL DEPTHWISE ATTENTION BLOCK
# ============================================================

class ResidualDepthwiseAttentionBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dropout_rate=0.0
    ):

        super().__init__()

        self.conv1 = DepthwiseSeparableConv(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride
        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=out_channels,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            )
        )

        self.attention = CombinedAttention(
            out_channels
        )

        if dropout_rate > 0:

            self.dropout = nn.Dropout2d(
                p=dropout_rate
            )

        else:

            self.dropout = nn.Identity()

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm2d(
                    out_channels
                )
            )

        else:

            self.shortcut = nn.Identity()

        self.activation = nn.SiLU(
            inplace=True
        )

    def forward(self, x):

        identity = self.shortcut(x)

        output = self.conv1(x)
        output = self.conv2(output)

        output = self.attention(output)
        output = self.dropout(output)

        output = output + identity
        output = self.activation(output)

        return output


# ============================================================
# CUSTOM FD-NET MODEL
# ============================================================

class FDNet(nn.Module):

    def __init__(
        self,
        num_classes=4
    ):

        super().__init__()

        # Initial feature extraction
        self.stem = nn.Sequential(

            ConvBNActivation(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=2
            ),

            ConvBNActivation(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1
            )
        )

        # Stage 1: output size approximately 112 × 112
        self.stage1 = nn.Sequential(

            ResidualDepthwiseAttentionBlock(
                in_channels=32,
                out_channels=32,
                stride=1,
                dropout_rate=0.05
            ),

            ResidualDepthwiseAttentionBlock(
                in_channels=32,
                out_channels=32,
                stride=1,
                dropout_rate=0.05
            )
        )

        # Stage 2: output size approximately 56 × 56
        self.stage2 = nn.Sequential(

            ResidualDepthwiseAttentionBlock(
                in_channels=32,
                out_channels=64,
                stride=2,
                dropout_rate=0.05
            ),

            ResidualDepthwiseAttentionBlock(
                in_channels=64,
                out_channels=64,
                stride=1,
                dropout_rate=0.05
            )
        )

        # Stage 3: output size approximately 28 × 28
        self.stage3 = nn.Sequential(

            ResidualDepthwiseAttentionBlock(
                in_channels=64,
                out_channels=128,
                stride=2,
                dropout_rate=0.10
            ),

            ResidualDepthwiseAttentionBlock(
                in_channels=128,
                out_channels=128,
                stride=1,
                dropout_rate=0.10
            )
        )

        # Stage 4: output size approximately 14 × 14
        self.stage4 = nn.Sequential(

            ResidualDepthwiseAttentionBlock(
                in_channels=128,
                out_channels=256,
                stride=2,
                dropout_rate=0.10
            ),

            ResidualDepthwiseAttentionBlock(
                in_channels=256,
                out_channels=256,
                stride=1,
                dropout_rate=0.10
            )
        )

        # Final feature expansion
        self.final_features = nn.Sequential(

            ConvBNActivation(
                in_channels=256,
                out_channels=384,
                kernel_size=1,
                stride=1
            ),

            CombinedAttention(
                channels=384
            )
        )

        # Classification head
        self.global_average_pooling = nn.AdaptiveAvgPool2d(
            output_size=1
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(
                p=0.40
            ),

            nn.Linear(
                in_features=384,
                out_features=128
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Dropout(
                p=0.30
            ),

            nn.Linear(
                in_features=128,
                out_features=num_classes
            )
        )

        self._initialize_weights()

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(module, nn.Conv2d):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

            elif isinstance(module, nn.BatchNorm2d):

                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

            elif isinstance(module, nn.BatchNorm1d):

                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

            elif isinstance(module, nn.Linear):

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.01
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

    def forward(self, x):

        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.final_features(x)

        x = self.global_average_pooling(x)
        x = self.classifier(x)

        return x


# ============================================================
# CREATE MODEL
# ============================================================

model = FDNet(
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# PARAMETER COUNT
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print("\nModel Created Successfully")

print("Model: FD-Net")

print(
    f"Total parameters    : "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)

print(
    f"Number of classes   : "
    f"{NUM_CLASSES}"
)


# ============================================================
# TEST MODEL OUTPUT SHAPE
# ============================================================

model.eval()

with torch.no_grad():

    dummy_input = torch.randn(
        2,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    ).to(DEVICE)

    dummy_output = model(
        dummy_input
    )

print(
    f"Test input shape : "
    f"{tuple(dummy_input.shape)}"
)

print(
    f"Test output shape: "
    f"{tuple(dummy_output.shape)}"
)

if dummy_output.shape != (2, NUM_CLASSES):

    raise RuntimeError(
        "FD-Net output shape is incorrect."
    )


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)


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

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5,
    min_lr=1e-7
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
    "Dataset, FD-Net architecture, optimizer "
    "and scheduler are ready."
)
# ============================================================
# FD-NET FABRIC DEFECT CLASSIFICATION
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
# TRAINING AND VALIDATION
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
    # SAVE HISTORY
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
    # SAVE HISTORY CSV
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
                train_dataset.class_to_idx,

            "total_parameters":
                total_parameters,

            "trainable_parameters":
                trainable_parameters
        }

        torch.save(
            checkpoint,
            MODEL_SAVE_PATH
        )

        print(
            "Best FD-Net model saved successfully."
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
# SAVE FINAL HISTORY
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

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("FD-Net Training and Validation Loss")

plt.legend()
plt.grid(True, alpha=0.3)
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

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("FD-Net Training and Validation Accuracy")

plt.legend()
plt.grid(True, alpha=0.3)
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

plt.xlabel("Epoch")
plt.ylabel("F1-score")
plt.title("FD-Net Validation F1-score")

plt.legend()
plt.grid(True, alpha=0.3)
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

plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("FD-Net Learning Rate Schedule")

plt.legend()
plt.grid(True, alpha=0.3)
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
    "\nLoading Best FD-Net Model"
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

print(
    f"Total Parameters: "
    f"{checkpoint['total_parameters']:,}"
)


# ============================================================
# TESTING
# ============================================================

print(
    "\nTesting Best FD-Net Model"
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
# TEST METRICS
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


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL FD-NET TEST RESULTS")
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
# SAVE TEXT RESULTS
# ============================================================

with open(
    RESULT_DIR / "test_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "FD-NET FABRIC DEFECT CLASSIFICATION\n"
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
        f"Total Parameters: "
        f"{checkpoint['total_parameters']:,}\n"
    )

    file.write(
        f"Trainable Parameters: "
        f"{checkpoint['trainable_parameters']:,}\n"
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
        "FD-Net",

    "Architecture":
        (
            "Residual + Depthwise Separable Convolution "
            "+ Channel Attention + Spatial Attention"
        ),

    "Best_Epoch":
        checkpoint["epoch"],

    "Best_Validation_F1":
        checkpoint["best_valid_f1"],

    "Total_Parameters":
        checkpoint["total_parameters"],

    "Trainable_Parameters":
        checkpoint["trainable_parameters"],

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
# SAVE TEST PREDICTIONS
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

    image_path = test_dataset.samples[
        sample_index
    ][0]

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

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("FD-Net Confusion Matrix")

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

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("FD-Net Normalized Confusion Matrix")

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "normalized_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE MODEL ARCHITECTURE
# ============================================================

with open(
    RESULT_DIR / "model_architecture.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(str(model))

    file.write("\n\n")

    file.write(
        f"Total Parameters: "
        f"{total_parameters:,}\n"
    )

    file.write(
        f"Trainable Parameters: "
        f"{trainable_parameters:,}\n"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\nAll files saved successfully at:")

print(
    RESULT_DIR
)