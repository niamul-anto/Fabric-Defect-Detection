# ============================================================
# ViT-B/16 Fabric Defect Classification
# Complete Paper Training Script - Part 1
# ============================================================


import os
import json
import random
from pathlib import Path
from collections import Counter


import numpy as np
import pandas as pd

import torch
from torch import nn

from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler
)

from torchvision import datasets, transforms
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights
)


from tqdm import tqdm



# ============================================================
# SEED
# ============================================================

SEED = 42


def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)



seed_everything(SEED)



# ============================================================
# PATH
# ============================================================


ROOT = Path(__file__).resolve().parent


DATA_DIR = (
    ROOT /
    "classification_dataset"
)


TRAIN_DIR = DATA_DIR / "train"

VALID_DIR = DATA_DIR / "valid"

TEST_DIR = DATA_DIR / "test"



RESULT_DIR = (
    ROOT /
    "results" /
    "vit_classification"
)


RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



MODEL_SAVE = (
    RESULT_DIR /
    "best_vit.pt"
)



# ============================================================
# SETTINGS
# ============================================================


IMAGE_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 100

LR = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 15



CLASSES = [

    "Cut",
    "Hole",
    "Stain",
    "ThreadError"

]



# ============================================================
# DEVICE
# ============================================================


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



print("="*70)

print(
    "ViT-B/16 FABRIC DEFECT CLASSIFICATION"
)

print("="*70)


print(
    "Device:",
    DEVICE
)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )



# ============================================================
# TRANSFORMS
# ============================================================


weights = (
    ViT_B_16_Weights.IMAGENET1K_V1
)



mean = weights.transforms().mean

std = weights.transforms().std



train_transform = transforms.Compose([


    transforms.Resize(
        (256,256)
    ),


    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8,1.0)
    ),


    transforms.RandomHorizontalFlip(
        p=0.5
    ),


    transforms.RandomRotation(
        10
    ),


    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),


    transforms.ToTensor(),


    transforms.Normalize(
        mean,
        std
    )

])



test_transform = transforms.Compose([


    transforms.Resize(
        (IMAGE_SIZE,IMAGE_SIZE)
    ),


    transforms.ToTensor(),


    transforms.Normalize(
        mean,
        std
    )

])



# ============================================================
# DATASET
# ============================================================



train_dataset = datasets.ImageFolder(

    TRAIN_DIR,

    transform=train_transform

)



valid_dataset = datasets.ImageFolder(

    VALID_DIR,

    transform=test_transform

)



test_dataset = datasets.ImageFolder(

    TEST_DIR,

    transform=test_transform

)



print("\nClass Mapping:")

print(
    train_dataset.class_to_idx
)



# ============================================================
# CLASS BALANCE
# ============================================================



targets = train_dataset.targets


class_count = Counter(targets)



print("\nTraining Distribution")

for idx,count in class_count.items():

    print(
        train_dataset.classes[idx],
        ":",
        count
    )



class_weights = []

total = len(targets)



for i in range(len(CLASSES)):

    class_weights.append(

        total /
        (
            len(CLASSES)
            *
            class_count[i]
        )

    )



class_weights = torch.tensor(

    class_weights,

    dtype=torch.float

)



sample_weights = []


for label in targets:

    sample_weights.append(

        class_weights[label]

    )



sampler = WeightedRandomSampler(

    sample_weights,

    len(sample_weights),

    replacement=True

)



# ============================================================
# DATALOADER
# ============================================================


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    sampler=sampler,

    num_workers=0

)



valid_loader = DataLoader(

    valid_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)



test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)



print("\nDataset Ready")

print(
    "Train:",
    len(train_dataset)
)

print(
    "Valid:",
    len(valid_dataset)
)

print(
    "Test:",
    len(test_dataset)
)



# ============================================================
# MODEL
# ============================================================



model = vit_b_16(

    weights=weights

)



model.heads.head = nn.Linear(

    model.heads.head.in_features,

    len(CLASSES)

)



model = model.to(DEVICE)



criterion = nn.CrossEntropyLoss()



optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LR,

    weight_decay=WEIGHT_DECAY

)



scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=5

)



print("\nModel Loaded Successfully")
# ============================================================
# TRAINING + EVALUATION
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



# ============================================================
# HISTORY
# ============================================================


history = {

    "epoch": [],

    "train_loss": [],

    "valid_loss": [],

    "train_accuracy": [],

    "valid_accuracy": [],

    "valid_f1": []

}



best_f1 = 0.0

patience_counter = 0



# ============================================================
# TRAIN LOOP
# ============================================================


for epoch in range(EPOCHS):


    print("\n")
    print("="*60)

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
    )

    print("="*60)



    # -------------------------
    # TRAIN
    # -------------------------


    model.train()


    train_loss = 0

    train_pred = []

    train_true = []



    for images, labels in tqdm(
        train_loader,
        desc="Training"
    ):


        images = images.to(DEVICE)

        labels = labels.to(DEVICE)



        optimizer.zero_grad()



        outputs = model(images)



        loss = criterion(
            outputs,
            labels
        )



        loss.backward()


        optimizer.step()



        train_loss += loss.item()



        preds = torch.argmax(
            outputs,
            dim=1
        )



        train_pred.extend(
            preds.cpu().numpy()
        )


        train_true.extend(
            labels.cpu().numpy()
        )



    train_loss /= len(train_loader)



    train_acc = accuracy_score(

        train_true,

        train_pred

    )




    # -------------------------
    # VALIDATION
    # -------------------------


    model.eval()


    valid_loss = 0


    valid_pred = []

    valid_true = []



    with torch.no_grad():


        for images, labels in tqdm(

            valid_loader,

            desc="Validation"

        ):


            images = images.to(DEVICE)

            labels = labels.to(DEVICE)



            outputs = model(images)



            loss = criterion(

                outputs,

                labels

            )



            valid_loss += loss.item()



            preds = torch.argmax(

                outputs,

                dim=1

            )



            valid_pred.extend(

                preds.cpu().numpy()

            )


            valid_true.extend(

                labels.cpu().numpy()

            )




    valid_loss /= len(valid_loader)



    valid_acc = accuracy_score(

        valid_true,

        valid_pred

    )



    valid_f1 = f1_score(

        valid_true,

        valid_pred,

        average="macro"

    )



    scheduler.step(valid_f1)



    print(
        f"Train Loss : {train_loss:.4f}"
    )


    print(
        f"Valid Loss : {valid_loss:.4f}"
    )


    print(
        f"Train Acc  : {train_acc:.4f}"
    )


    print(
        f"Valid Acc  : {valid_acc:.4f}"
    )


    print(
        f"Valid F1   : {valid_f1:.4f}"
    )



    # save history


    history["epoch"].append(

        epoch+1

    )


    history["train_loss"].append(

        train_loss

    )


    history["valid_loss"].append(

        valid_loss

    )


    history["train_accuracy"].append(

        train_acc

    )


    history["valid_accuracy"].append(

        valid_acc

    )


    history["valid_f1"].append(

        valid_f1

    )



    # -------------------------
    # SAVE BEST MODEL
    # -------------------------


    if valid_f1 > best_f1:


        best_f1 = valid_f1


        patience_counter = 0



        torch.save(

            model.state_dict(),

            MODEL_SAVE

        )


        print(
            "Best model saved"
        )



    else:


        patience_counter += 1



    if patience_counter >= PATIENCE:


        print(
            "Early stopping"
        )

        break





# ============================================================
# SAVE TRAINING HISTORY
# ============================================================


history_df = pd.DataFrame(history)


history_df.to_csv(

    RESULT_DIR /
    "training_history.csv",

    index=False

)



print("\nTraining Completed")



# ============================================================
# LOSS CURVE
# ============================================================


plt.figure(
    figsize=(8,5)
)


plt.plot(

    history["epoch"],

    history["train_loss"],

    label="Train Loss"

)


plt.plot(

    history["epoch"],

    history["valid_loss"],

    label="Validation Loss"

)


plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "ViT-B/16 Loss Curve"
)


plt.legend()

plt.grid()



plt.savefig(

    RESULT_DIR /
    "loss_curve.png",

    dpi=300,

    bbox_inches="tight"

)


plt.close()



# ============================================================
# ACCURACY CURVE
# ============================================================


plt.figure(

    figsize=(8,5)

)



plt.plot(

    history["epoch"],

    history["train_accuracy"],

    label="Train Accuracy"

)



plt.plot(

    history["epoch"],

    history["valid_accuracy"],

    label="Validation Accuracy"

)



plt.xlabel("Epoch")

plt.ylabel("Accuracy")


plt.title(
    "ViT-B/16 Accuracy Curve"
)


plt.legend()


plt.grid()



plt.savefig(

    RESULT_DIR /
    "accuracy_curve.png",

    dpi=300,

    bbox_inches="tight"

)


plt.close()





# ============================================================
# TEST
# ============================================================



print("\nTesting Best Model")



model.load_state_dict(

    torch.load(

        MODEL_SAVE,

        map_location=DEVICE

    )

)



model.eval()



test_true=[]

test_pred=[]



with torch.no_grad():


    for images, labels in tqdm(

        test_loader,

        desc="Testing"

    ):


        images = images.to(DEVICE)



        outputs = model(images)



        preds = torch.argmax(

            outputs,

            dim=1

        )



        test_pred.extend(

            preds.cpu().numpy()

        )


        test_true.extend(

            labels.numpy()

        )




# ============================================================
# FINAL METRICS
# ============================================================


accuracy = accuracy_score(

    test_true,

    test_pred

)


precision = precision_score(

    test_true,

    test_pred,

    average="macro"

)


recall = recall_score(

    test_true,

    test_pred,

    average="macro"

)


f1 = f1_score(

    test_true,

    test_pred,

    average="macro"

)



print("\n")
print("="*60)

print("FINAL TEST RESULTS")

print("="*60)


print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1-score : {f1:.4f}"
)



# save text


with open(

    RESULT_DIR /
    "test_results.txt",

    "w"

) as f:


    f.write(
        f"Accuracy : {accuracy:.6f}\n"
    )

    f.write(
        f"Precision: {precision:.6f}\n"
    )

    f.write(
        f"Recall   : {recall:.6f}\n"
    )

    f.write(
        f"F1-score : {f1:.6f}\n"
    )




# ============================================================
# CLASSIFICATION REPORT
# ============================================================


report = classification_report(

    test_true,

    test_pred,

    target_names=CLASSES,

    output_dict=True

)



pd.DataFrame(report).transpose().to_csv(

    RESULT_DIR /
    "classification_report.csv"

)



# ============================================================
# CONFUSION MATRIX
# ============================================================


cm = confusion_matrix(

    test_true,

    test_pred

)



plt.figure(

    figsize=(8,7)

)



sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=CLASSES,

    yticklabels=CLASSES

)



plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.title(
    "ViT-B/16 Confusion Matrix"
)



plt.tight_layout()



plt.savefig(

    RESULT_DIR /
    "confusion_matrix.png",

    dpi=300

)



plt.close()



# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================


cm_norm = (

    cm.astype(float)

    /

    cm.sum(axis=1)[:,None]

)



plt.figure(

    figsize=(8,7)

)



sns.heatmap(

    cm_norm,

    annot=True,

    fmt=".2f",

    cmap="Blues",

    xticklabels=CLASSES,

    yticklabels=CLASSES

)



plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.title(
    "Normalized Confusion Matrix"
)



plt.tight_layout()



plt.savefig(

    RESULT_DIR /
    "normalized_confusion_matrix.png",

    dpi=300

)



plt.close()



print("\nAll files saved at:")

print(RESULT_DIR)