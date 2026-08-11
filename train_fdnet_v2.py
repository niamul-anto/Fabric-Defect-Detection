# ============================================================
# FD-NET V2
# PART 1: DATASET, MODEL ARCHITECTURE AND TRAINING SETUP
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
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "classification_dataset"

TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"

RESULT_DIR = ROOT_DIR / "results" / "fdnet_v2_classification"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = RESULT_DIR / "best_fdnet_v2.pt"


# ============================================================
# CONFIGURATION
# ============================================================

CLASSES = [
    "Cut",
    "Hole",
    "Stain",
    "ThreadError"
]

NUM_CLASSES = len(CLASSES)

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 100

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 20
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("FD-NET V2 FABRIC DEFECT CLASSIFICATION")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# DATA AUGMENTATION
# ============================================================

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.85, 1.00),
        ratio=(0.90, 1.10)
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(degrees=7),

    transforms.ColorJitter(
        brightness=0.12,
        contrast=0.15,
        saturation=0.08,
        hue=0.01
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )
])

valid_test_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

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
    TRAIN_DIR,
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(
    VALID_DIR,
    transform=valid_test_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=valid_test_transform
)


# ============================================================
# VERIFY CLASS MAPPING
# ============================================================

expected_mapping = {
    class_name: index
    for index, class_name in enumerate(CLASSES)
}

print("\nDetected class mapping:")
print(train_dataset.class_to_idx)

if train_dataset.class_to_idx != expected_mapping:
    raise ValueError(
        f"Class mapping mismatch.\n"
        f"Detected: {train_dataset.class_to_idx}\n"
        f"Expected: {expected_mapping}"
    )

if valid_dataset.class_to_idx != expected_mapping:
    raise ValueError("Validation class mapping mismatch.")

if test_dataset.class_to_idx != expected_mapping:
    raise ValueError("Test class mapping mismatch.")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

train_class_counts = Counter(train_dataset.targets)

print("\nTraining distribution:")

for class_index, class_name in enumerate(CLASSES):
    print(
        f"{class_name:<15}: "
        f"{train_class_counts[class_index]}"
    )

print("\nDataset sizes:")

print(f"Training   : {len(train_dataset)}")
print(f"Validation : {len(valid_dataset)}")
print(f"Testing    : {len(test_dataset)}")


# ============================================================
# WEIGHTED RANDOM SAMPLER
# ============================================================

total_samples = len(train_dataset)

class_sampling_weights = {}

for class_index in range(NUM_CLASSES):

    class_count = train_class_counts[class_index]

    if class_count == 0:
        raise ValueError(
            f"No samples found for {CLASSES[class_index]}"
        )

    class_sampling_weights[class_index] = (
        total_samples /
        (NUM_CLASSES * class_count)
    )

sample_weights = [
    class_sampling_weights[label]
    for label in train_dataset.targets
]

train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

print("\nSampling weights:")

for class_index, class_name in enumerate(CLASSES):
    print(
        f"{class_name:<15}: "
        f"{class_sampling_weights[class_index]:.6f}"
    )


# ============================================================
# DATA LOADERS
# ============================================================

pin_memory = DEVICE.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)

print("\nDataLoader batches:")

print(f"Training   : {len(train_loader)}")
print(f"Validation : {len(valid_loader)}")
print(f"Testing    : {len(test_loader)}")


# ============================================================
# STANDARD CONVOLUTION BLOCK
# ============================================================

class ConvBNAct(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
        activation=True
    ):

        super().__init__()

        padding = kernel_size // 2

        layers = [

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False
            ),

            nn.BatchNorm2d(out_channels)
        ]

        if activation:
            layers.append(
                nn.SiLU(inplace=True)
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ============================================================
# CHANNEL ATTENTION
# ============================================================

class ChannelAttention(nn.Module):

    def __init__(
        self,
        channels,
        reduction=16
    ):

        super().__init__()

        hidden_channels = max(
            channels // reduction,
            16
        )

        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.maximum_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_network = nn.Sequential(

            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False
            ),

            nn.SiLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
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

    def __init__(self, kernel_size=7):

        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
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
# CHANNEL + SPATIAL ATTENTION
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
# INVERTED RESIDUAL DEPTHWISE BLOCK
# ============================================================

class InvertedResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expansion=4,
        use_attention=False,
        dropout_rate=0.0
    ):

        super().__init__()

        hidden_channels = in_channels * expansion

        self.use_residual = (
            stride == 1 and
            in_channels == out_channels
        )

        # 1 × 1 channel expansion
        self.expand = ConvBNAct(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1
        )

        # 3 × 3 depthwise convolution
        self.depthwise = ConvBNAct(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=stride,
            groups=hidden_channels
        )

        # Channel-spatial attention only in later stages
        if use_attention:
            self.attention = CombinedAttention(
                hidden_channels
            )
        else:
            self.attention = nn.Identity()

        # 1 × 1 projection
        self.project = ConvBNAct(
            hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            activation=False
        )

        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(
                p=dropout_rate
            )
        else:
            self.dropout = nn.Identity()

        # Shortcut for dimension changes
        if stride != 1 or in_channels != out_channels:

            self.shortcut = ConvBNAct(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                activation=False
            )

        else:
            self.shortcut = nn.Identity()

        self.activation = nn.SiLU(
            inplace=True
        )

    def forward(self, x):

        identity = self.shortcut(x)

        output = self.expand(x)
        output = self.depthwise(output)
        output = self.attention(output)
        output = self.project(output)
        output = self.dropout(output)

        output = output + identity
        output = self.activation(output)

        return output


# ============================================================
# FD-NET V2
# ============================================================

class FDNetV2(nn.Module):

    def __init__(
        self,
        num_classes=4
    ):

        super().__init__()

        # ----------------------------------------------------
        # Stem: 224 -> 112
        # ----------------------------------------------------

        self.stem = nn.Sequential(

            ConvBNAct(
                3,
                48,
                kernel_size=3,
                stride=2
            ),

            ConvBNAct(
                48,
                48,
                kernel_size=3,
                stride=1
            )
        )

        # ----------------------------------------------------
        # Stage 1: standard convolution features
        # 112 -> 56
        # ----------------------------------------------------

        self.stage1 = nn.Sequential(

            ConvBNAct(
                48,
                64,
                kernel_size=3,
                stride=2
            ),

            ConvBNAct(
                64,
                64,
                kernel_size=3,
                stride=1
            )
        )

        # ----------------------------------------------------
        # Stage 2: residual depthwise blocks
        # 56 -> 28
        # No attention
        # ----------------------------------------------------

        self.stage2 = nn.Sequential(

            InvertedResidualBlock(
                64,
                96,
                stride=2,
                expansion=3,
                use_attention=False
            ),

            InvertedResidualBlock(
                96,
                96,
                stride=1,
                expansion=3,
                use_attention=False
            ),

            InvertedResidualBlock(
                96,
                96,
                stride=1,
                expansion=3,
                use_attention=False
            )
        )

        # ----------------------------------------------------
        # Stage 3: residual + attention
        # 28 -> 14
        # ----------------------------------------------------

        self.stage3 = nn.Sequential(

            InvertedResidualBlock(
                96,
                192,
                stride=2,
                expansion=4,
                use_attention=True,
                dropout_rate=0.03
            ),

            InvertedResidualBlock(
                192,
                192,
                stride=1,
                expansion=4,
                use_attention=True,
                dropout_rate=0.03
            ),

            InvertedResidualBlock(
                192,
                192,
                stride=1,
                expansion=4,
                use_attention=True,
                dropout_rate=0.03
            )
        )

        # ----------------------------------------------------
        # Stage 4: residual + attention
        # 14 -> 7
        # ----------------------------------------------------

        self.stage4 = nn.Sequential(

            InvertedResidualBlock(
                192,
                320,
                stride=2,
                expansion=4,
                use_attention=True,
                dropout_rate=0.05
            ),

            InvertedResidualBlock(
                320,
                320,
                stride=1,
                expansion=4,
                use_attention=True,
                dropout_rate=0.05
            )
        )

        # ----------------------------------------------------
        # Final feature layer
        # ----------------------------------------------------

        self.final_features = nn.Sequential(

            ConvBNAct(
                320,
                512,
                kernel_size=1,
                stride=1
            ),

            CombinedAttention(512)
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(p=0.25),

            nn.Linear(
                512,
                256
            ),

            nn.BatchNorm1d(256),

            nn.SiLU(inplace=True),

            nn.Dropout(p=0.20),

            nn.Linear(
                256,
                num_classes
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

            elif isinstance(
                module,
                (nn.BatchNorm2d, nn.BatchNorm1d)
            ):

                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.01
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):

        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.final_features(x)

        x = self.global_pool(x)
        x = self.classifier(x)

        return x


# ============================================================
# CREATE MODEL
# ============================================================

model = FDNetV2(
    num_classes=NUM_CLASSES
).to(DEVICE)


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

print("\nModel created successfully")

print("Model: FD-Net V2")

print(
    f"Total parameters    : "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)


# ============================================================
# OUTPUT SHAPE TEST
# ============================================================

model.eval()

with torch.no_grad():

    dummy_input = torch.randn(
        2,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE
    )

    dummy_output = model(dummy_input)

print(
    f"Input shape : "
    f"{tuple(dummy_input.shape)}"
)

print(
    f"Output shape: "
    f"{tuple(dummy_output.shape)}"
)

if dummy_output.shape != (2, NUM_CLASSES):
    raise RuntimeError(
        "Incorrect model output shape."
    )


# ============================================================
# LOSS
# No label smoothing
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
# COSINE ANNEALING SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)


# ============================================================
# MIXED PRECISION
# ============================================================

scaler = torch.amp.GradScaler(
    device=DEVICE.type,
    enabled=DEVICE.type == "cuda"
)


# ============================================================
# HISTORY
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
print("FD-Net V2 is ready for training.")
# ============================================================
# FD-NET V2
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

    # CosineAnnealingLR does not use validation metric
    scheduler.step()

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

            "scheduler_state_dict":
                scheduler.state_dict(),

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
            "Best FD-Net V2 model saved successfully."
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
plt.title("FD-Net V2 Training and Validation Loss")

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
plt.title("FD-Net V2 Training and Validation Accuracy")

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
plt.title("FD-Net V2 Validation F1-score")

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
plt.title("FD-Net V2 Learning Rate Schedule")

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
    "\nLoading Best FD-Net V2 Model"
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
    "\nTesting Best FD-Net V2 Model"
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
print("FINAL FD-NET V2 TEST RESULTS")
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
        "FD-NET V2 FABRIC DEFECT CLASSIFICATION\n"
    )

    file.write(
        "=" * 60 + "\n"
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
        "FD-Net V2",

    "Architecture":
        (
            "Residual Inverted Bottleneck "
            "+ Depthwise Convolution "
            "+ Channel Attention "
            "+ Spatial Attention"
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
plt.title("FD-Net V2 Confusion Matrix")

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
plt.title("FD-Net V2 Normalized Confusion Matrix")

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
