# ============================================================
# Faster R-CNN Fabric Defect Detection
# Part 1 : Setup + Dataset + Model + Training
# ============================================================

import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from torchvision.transforms import functional as TF

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights
)

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

DATASET = ROOT / "yolo_dataset"

SAVE_DIR = ROOT / "results" / "fasterrcnn_fabric"


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


CLASSES = [
    "background",
    "Cut",
    "Hole",
    "Stain",
    "ThreadError"
]


NUM_CLASSES = len(CLASSES)


EPOCHS = 150

BATCH_SIZE = 4

NUM_WORKERS = 0

LR = 0.001

WEIGHT_DECAY = 0.0005

MOMENTUM = 0.9

PATIENCE = 30


IMAGE_MIN = 960
IMAGE_MAX = 1280



# ============================================================
# SEED
# ============================================================

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# ============================================================
# DATASET
# ============================================================

class FabricDataset(Dataset):

    def __init__(
            self,
            folder,
            train=False):

        self.image_dir = folder / "images"
        self.label_dir = folder / "labels"

        self.train = train


        self.images = sorted(
            [
                x for x in self.image_dir.iterdir()
                if x.suffix.lower()
                in [
                    ".jpg",
                    ".jpeg",
                    ".png"
                ]
            ]
        )


    def __len__(self):

        return len(self.images)



    def read_labels(
            self,
            file,
            width,
            height):

        boxes = []
        labels = []


        if file.exists():

            lines = file.read_text().strip().splitlines()


            for line in lines:

                if not line:
                    continue


                cls,x,y,w,h = map(
                    float,
                    line.split()
                )


                x *= width
                y *= height
                w *= width
                h *= height


                xmin = x - w/2
                ymin = y - h/2

                xmax = x + w/2
                ymax = y + h/2


                boxes.append(
                    [
                        xmin,
                        ymin,
                        xmax,
                        ymax
                    ]
                )


                # Faster R-CNN:
                # 0 = background
                labels.append(
                    int(cls)+1
                )


        return (

            torch.tensor(
                boxes,
                dtype=torch.float32
            ),

            torch.tensor(
                labels,
                dtype=torch.int64
            )
        )



    def __getitem__(self,index):

        img_path = self.images[index]


        image = Image.open(
            img_path
        ).convert("RGB")


        width,height = image.size


        label_path = (
            self.label_dir /
            f"{img_path.stem}.txt"
        )


        boxes,labels = self.read_labels(
            label_path,
            width,
            height
        )


        # augmentation

        if self.train:

            if random.random() < 0.5:

                image = TF.hflip(image)


                if len(boxes) > 0:

                    old_x1 = boxes[:,0].clone()
                    old_x2 = boxes[:,2].clone()


                    boxes[:,0] = (
                        width - old_x2
                    )

                    boxes[:,2] = (
                        width - old_x1
                    )



        image = TF.to_tensor(image)



        target = {

            "boxes": boxes,

            "labels": labels,

            "image_id":
                torch.tensor([index]),


            "area":
                (
                    boxes[:,2]-boxes[:,0]
                )
                *
                (
                    boxes[:,3]-boxes[:,1]
                ),


            "iscrowd":
                torch.zeros(
                    len(labels),
                    dtype=torch.int64
                )
        }


        return image,target



def collate_fn(batch):

    return tuple(zip(*batch))



# ============================================================
# MODEL
# ============================================================

def create_model():

    weights = (
        FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )


    model = fasterrcnn_resnet50_fpn_v2(

        weights=weights,

        min_size=IMAGE_MIN,

        max_size=IMAGE_MAX,

        box_detections_per_img=200

    )


    in_features = (
        model.roi_heads
        .box_predictor
        .cls_score
        .in_features
    )


    model.roi_heads.box_predictor = (

        FastRCNNPredictor(
            in_features,
            NUM_CLASSES
        )

    )


    return model



# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
        model,
        loader,
        optimizer):


    model.train()


    total_loss = 0



    progress = tqdm(
        loader,
        desc="Training"
    )


    for images,targets in progress:


        images = [
            img.to(DEVICE)
            for img in images
        ]


        targets = [

            {
                k:v.to(DEVICE)
                for k,v in target.items()
            }

            for target in targets

        ]



        loss_dict = model(
            images,
            targets
        )


        loss = sum(
            loss for loss in loss_dict.values()
        )


        optimizer.zero_grad()


        loss.backward()


        optimizer.step()



        total_loss += loss.item()



        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )



    return total_loss / len(loader)
# ============================================================
# Part 2 : Evaluation + Graph + Main
# ============================================================

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)


from torchvision.ops import box_iou



# ============================================================
# EVALUATION (COCO mAP)
# ============================================================

@torch.no_grad()
def evaluate_map(model, loader):

    model.eval()


    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox"
    )


    for images, targets in tqdm(
        loader,
        desc="Evaluating"
    ):

        images = [
            img.to(DEVICE)
            for img in images
        ]


        outputs = model(images)


        preds = []
        gts = []


        for output,target in zip(
            outputs,
            targets
        ):


            preds.append({

                "boxes":
                    output["boxes"].cpu(),

                "scores":
                    output["scores"].cpu(),

                "labels":
                    output["labels"].cpu()

            })


            gts.append({

                "boxes":
                    target["boxes"],

                "labels":
                    target["labels"]

            })


        metric.update(
            preds,
            gts
        )


    return metric.compute()



# ============================================================
# CONFUSION MATRIX
# ============================================================


@torch.no_grad()
def create_confusion_matrix(
        model,
        loader,
        threshold=0.5):


    model.eval()


    y_true=[]
    y_pred=[]


    for images,targets in tqdm(
        loader,
        desc="Confusion Matrix"
    ):


        images=[
            img.to(DEVICE)
            for img in images
        ]


        outputs=model(images)



        for output,target in zip(
            outputs,
            targets
        ):


            gt_labels = (
                target["labels"]
                .numpy()
                .tolist()
            )


            pred_labels=[]


            for score,label in zip(
                output["scores"],
                output["labels"]
            ):

                if score >= threshold:

                    pred_labels.append(
                        label.item()
                    )


            length=max(
                len(gt_labels),
                len(pred_labels)
            )


            gt_labels += (
                [0] *
                (length-len(gt_labels))
            )


            pred_labels += (
                [0] *
                (length-len(pred_labels))
            )


            y_true.extend(
                gt_labels
            )

            y_pred.extend(
                pred_labels
            )



    matrix = confusion_matrix(

        y_true,

        y_pred,

        labels=list(
            range(NUM_CLASSES)
        )

    )


    return matrix



# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================


def save_confusion(matrix):


    plt.figure(
        figsize=(8,8)
    )


    disp = ConfusionMatrixDisplay(

        confusion_matrix=matrix,

        display_labels=CLASSES

    )


    disp.plot(
        cmap="Blues",
        values_format="d"
    )


    plt.title(
        "Faster R-CNN Confusion Matrix"
    )


    plt.savefig(
        SAVE_DIR /
        "confusion_matrix.png",

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()



    # normalized


    normalized = (
        matrix /
        matrix.sum(
            axis=1,
            keepdims=True
        )
    )


    normalized = np.nan_to_num(
        normalized
    )


    plt.figure(
        figsize=(8,8)
    )


    disp = ConfusionMatrixDisplay(

        confusion_matrix=normalized,

        display_labels=CLASSES

    )


    disp.plot(

        cmap="Blues",

        values_format=".2f"

    )


    plt.title(
        "Normalized Confusion Matrix"
    )


    plt.savefig(

        SAVE_DIR /
        "confusion_matrix_normalized.png",

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



# ============================================================
# GRAPH
# ============================================================


def save_graph(history):


    df=pd.DataFrame(history)



    # Loss


    plt.figure(
        figsize=(8,5)
    )


    plt.plot(
        df["epoch"],
        df["loss"]
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.title(
        "Training Loss"
    )


    plt.grid()


    plt.savefig(

        SAVE_DIR /
        "loss_curve.png",

        dpi=300

    )


    plt.close()



    # mAP


    plt.figure(
        figsize=(8,5)
    )


    plt.plot(
        df["epoch"],
        df["map50"],
        label="mAP@0.5"
    )


    plt.plot(
        df["epoch"],
        df["map5095"],
        label="mAP@0.5:0.95"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "mAP"
    )


    plt.legend()


    plt.grid()


    plt.title(
        "Validation mAP"
    )


    plt.savefig(

        SAVE_DIR /
        "map_curve.png",

        dpi=300

    )


    plt.close()



# ============================================================
# MAIN
# ============================================================


def main():


    seed_everything()



    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print(
        "Device:",
        DEVICE
    )


    train_dataset = FabricDataset(

        DATASET/"train",

        train=True

    )


    valid_dataset = FabricDataset(

        DATASET/"valid"

    )


    test_dataset = FabricDataset(

        DATASET/"test"

    )



    train_loader=DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        collate_fn=collate_fn

    )



    valid_loader=DataLoader(

        valid_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        collate_fn=collate_fn

    )



    test_loader=DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        collate_fn=collate_fn

    )



    model=create_model()


    model.to(
        DEVICE
    )



    optimizer=torch.optim.SGD(

        model.parameters(),

        lr=LR,

        momentum=MOMENTUM,

        weight_decay=WEIGHT_DECAY

    )


    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=EPOCHS

    )



    history=[]


    best_map=0


    patience_count=0



    for epoch in range(1,EPOCHS+1):


        loss=train_one_epoch(

            model,

            train_loader,

            optimizer

        )


        val_result=evaluate_map(

            model,

            valid_loader

        )


        map50=float(
            val_result["map_50"]
        )


        map5095=float(
            val_result["map"]
        )


        scheduler.step()



        print(
f"""
Epoch {epoch}/{EPOCHS}

Loss:
{loss:.4f}

Val mAP50:
{map50:.4f}

Val mAP50-95:
{map5095:.4f}
"""
        )



        history.append({

            "epoch":epoch,

            "loss":loss,

            "map50":map50,

            "map5095":map5095

        })



        if map50 > best_map:


            best_map=map50

            patience_count=0


            torch.save(

                model.state_dict(),

                SAVE_DIR /
                "best_model.pth"

            )


            print(
                "Best model saved"
            )


        else:

            patience_count +=1



        if patience_count >= PATIENCE:


            print(
                "Early stopping"
            )

            break



    # save history


    with open(

        SAVE_DIR /
        "history.json",

        "w"

    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )


    save_graph(
        history
    )



    # =========================
    # TEST
    # =========================


    model.load_state_dict(

        torch.load(

            SAVE_DIR /
            "best_model.pth",

            map_location=DEVICE

        )

    )



    test_result=evaluate_map(

        model,

        test_loader

    )



    matrix=create_confusion_matrix(

        model,

        test_loader

    )


    save_confusion(
        matrix
    )



    precision=float(
        test_result["map_50"]
    )


    recall=float(
        test_result["mar_100"]
    )


    f1 = (
        2 *
        precision *
        recall /
        (precision+recall+1e-9)
    )



    results={

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "mAP50":
            float(
                test_result["map_50"]
            ),

        "mAP50-95":
            float(
                test_result["map"]
            )

    }



    pd.DataFrame(
        [results]
    ).to_csv(

        SAVE_DIR /
        "test_results.csv",

        index=False

    )



    print("\nFINAL TEST RESULTS")
    print("====================")


    for k,v in results.items():

        print(
            f"{k}: {v:.4f}"
        )



if __name__=="__main__":

    main()