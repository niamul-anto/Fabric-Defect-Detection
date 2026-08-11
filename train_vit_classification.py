import os
import json
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

from torchvision import datasets, transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from tqdm import tqdm


# ======================================================
# SETTINGS
# ======================================================


ROOT = Path(__file__).resolve().parent


DATA_DIR = ROOT / "classification_dataset"


TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR  = DATA_DIR / "test"


OUTPUT_DIR = ROOT / "results" / "vit_classification"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_PATH = OUTPUT_DIR / "best_vit.pt"


CLASSES = [
    "Cut",
    "Hole",
    "Stain",
    "ThreadError"
]


IMAGE_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 100

LR = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 15


SEED = 42



# ======================================================
# SEED
# ======================================================


def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# ======================================================
# TRANSFORM
# ======================================================


weights = ViT_B_16_Weights.IMAGENET1K_V1


train_transform = transforms.Compose([

    transforms.Resize(
        (256,256)
    ),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8,1.0)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        weights.transforms().mean,
        weights.transforms().std
    )

])


test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE,IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        weights.transforms().mean,
        weights.transforms().std
    )

])



# ======================================================
# DATASET
# ======================================================


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



print("\nClasses:")
print(train_dataset.class_to_idx)



# ======================================================
# CLASS WEIGHT
# ======================================================


targets = train_dataset.targets


class_counts = Counter(targets)


print("\nTraining distribution")

for idx,count in class_counts.items():

    print(
        train_dataset.classes[idx],
        ":",
        count
    )


weights_class = []


total = len(targets)


for i in range(len(CLASSES)):

    count = class_counts[i]

    weights_class.append(
        total/(len(CLASSES)*count)
    )


weights_class = torch.tensor(
    weights_class,
    dtype=torch.float
)



sample_weights = [

    weights_class[label]

    for label in targets

]


sampler = WeightedRandomSampler(

    sample_weights,

    len(sample_weights),

    replacement=True

)



# ======================================================
# LOADER
# ======================================================


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



# ======================================================
# MODEL
# ======================================================


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


print("\nDevice:",device)


model = vit_b_16(
    weights=weights
)


model.heads.head = nn.Linear(

    model.heads.head.in_features,

    len(CLASSES)

)


model = model.to(device)



criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LR,

    weight_decay=WEIGHT_DECAY

)



scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    patience=5,

    factor=0.5

)



# ======================================================
# TRAIN
# ======================================================


history = {

    "train_loss":[],
    "valid_loss":[],
    "train_acc":[],
    "valid_acc":[]

}



best_f1 = 0

counter = 0



for epoch in range(EPOCHS):


    print(
        f"\nEpoch {epoch+1}/{EPOCHS}"
    )


    model.train()


    total_loss = 0

    preds=[]

    labels=[]


    for x,y in tqdm(train_loader):


        x=x.to(device)

        y=y.to(device)



        optimizer.zero_grad()



        output=model(x)


        loss=criterion(
            output,
            y
        )


        loss.backward()


        optimizer.step()



        total_loss += loss.item()



        preds.extend(
            output.argmax(1)
            .cpu()
            .numpy()
        )


        labels.extend(
            y.cpu().numpy()
        )



    train_loss = total_loss/len(train_loader)


    train_acc = accuracy_score(
        labels,
        preds
    )



    # validation


    model.eval()


    vpred=[]

    vlabel=[]

    vloss=0



    with torch.no_grad():

        for x,y in valid_loader:


            x=x.to(device)

            y=y.to(device)


            output=model(x)


            loss=criterion(
                output,
                y
            )


            vloss+=loss.item()



            vpred.extend(
                output.argmax(1)
                .cpu()
                .numpy()
            )


            vlabel.extend(
                y.cpu().numpy()
            )



    valid_loss = vloss/len(valid_loader)


    valid_f1=f1_score(
        vlabel,
        vpred,
        average="macro"
    )


    valid_acc=accuracy_score(
        vlabel,
        vpred
    )


    scheduler.step(valid_f1)



    print(
        f"Train Acc: {train_acc:.4f}"
    )

    print(
        f"Valid Acc: {valid_acc:.4f}"
    )

    print(
        f"Valid F1 : {valid_f1:.4f}"
    )



    history["train_loss"].append(train_loss)

    history["valid_loss"].append(valid_loss)

    history["train_acc"].append(train_acc)

    history["valid_acc"].append(valid_acc)



    if valid_f1 > best_f1:


        best_f1=valid_f1

        counter=0


        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


        print("Best model saved")



    else:

        counter+=1



    if counter>=PATIENCE:

        print("Early stopping")

        break



# ======================================================
# TEST
# ======================================================


model.load_state_dict(
    torch.load(
        MODEL_PATH
    )
)


model.eval()


y_true=[]

y_pred=[]



with torch.no_grad():

    for x,y in tqdm(
        test_loader,
        desc="Testing"
    ):


        x=x.to(device)


        output=model(x)


        pred=output.argmax(1)


        y_pred.extend(
            pred.cpu().numpy()
        )


        y_true.extend(
            y.numpy()
        )



accuracy=accuracy_score(
    y_true,
    y_pred
)


precision=precision_score(
    y_true,
    y_pred,
    average="macro"
)


recall=recall_score(
    y_true,
    y_pred,
    average="macro"
)


f1=f1_score(
    y_true,
    y_pred,
    average="macro"
)



print("\n==============================")
print("FINAL TEST RESULT")
print("==============================")

print(
    "Accuracy:",
    accuracy
)

print(
    "Precision:",
    precision
)

print(
    "Recall:",
    recall
)

print(
    "F1:",
    f1
)



# report

report=classification_report(

    y_true,

    y_pred,

    target_names=CLASSES,

    output_dict=True

)


pd.DataFrame(report).transpose().to_csv(

    OUTPUT_DIR/"classification_report.csv"

)



cm=confusion_matrix(

    y_true,

    y_pred

)


plt.figure(figsize=(7,6))

plt.imshow(cm)

plt.colorbar()


plt.xticks(
    range(4),
    CLASSES,
    rotation=45
)

plt.yticks(
    range(4),
    CLASSES
)


plt.xlabel("Predicted")

plt.ylabel("True")

plt.title(
    "ViT Confusion Matrix"
)


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR/"confusion_matrix.png",
    dpi=300
)


print("\nSaved:")
print(OUTPUT_DIR)