# Fabric Defect Detection

A deep learning-based fabric defect detection and classification project using multiple computer vision models.

## Project Overview

This project focuses on detecting and classifying defects in fabric images using deep learning and computer vision techniques.

Multiple object detection and image classification models were trained and evaluated to compare their performance on the fabric defect dataset.

## Defect Classes

The dataset contains four major fabric defect classes:

1. Sewing Defects
2. Cotton Defects
3. Hole
4. Color Defects

## Models Used

The project includes training and evaluation of the following models:

### Object Detection Models
- YOLOv8n
- YOLOv8m
- RetinaNet
- Faster R-CNN

### Image Classification Models
- VGG16
- Vision Transformer (ViT)
- EfficientNet-B0
- FDNet
- FDNet-V2

## Project Structure

```text
Fabric-Defect-Detection/
│
├── results/
│   ├── efficientnet_b0_benchmark/
│   ├── efficientnet_b0_classification/
│   ├── fasterrcnn_fabric/
│   ├── fasterrcnn_resnet50_fpn_v2/
│   ├── fdnet_classification/
│   ├── fdnet_v2_benchmark/
│   ├── fdnet_v2_classification/
│   ├── retinanet_fabric/
│   ├── vgg16_benchmark/
│   ├── vgg16_classification/
│   ├── vit_benchmark/
│   ├── vit_classification/
│   ├── yolov8m_fabric_960/
│   ├── yolov8m_fabric_test/
│   ├── yolov8n_fabric/
│   └── yolov8n_test/
│
├── benchmark_efficientnet_b0.py
├── benchmark_fdnet_v2.py
├── benchmark_vgg16_bn.py
├── benchmark_vit_b16.py
│
├── convert_to_yolo.py
├── create_classification_dataset.py
│
├── evaluate_fasterrcnn.py
├── evaluate_retinanet.py
├── evaluate_yolov8m.py
├── evaluate_yolov8m_thresholds.py
│
├── train_efficientnet_b0_final.py
├── train_fasterrcnn.py
├── train_fdnet_final.py
├── train_fdnet_v2.py
├── train_retinanet.py
├── train_vgg16_classification.py
├── train_vit_b16_final.py
├── train_vit_classification.py
├── train_yolov8.py
├── train_yolov8m.py
│
├── data.yaml
├── README.dataset.txt
├── README.roboflow.txt
├── README.md
└── .gitignore
## Model Performance Comparison
### Classification Models

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 92.62% | 91.91% | 93.37% | 92.22% |
| FDNet | 62.70% | 62.25% | 61.55% | 60.54% |
| FDNet-V2 | 89.75% | 88.81% | 91.21% | 89.45% |
| VGG16-BN | 94.26% | 93.36% | 95.06% | 93.87% |
| **ViT** | **95.08%** | **94.28%** | **95.55%** | **94.84%** |

### Detection Models

| Model | Precision | Recall | F1-score | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 56.60% | 60.83% | — | 59.85% | 32.54% |
| YOLOv8m | 75.78% | 50.00% | 60.25% | 64.04% | 39.06% |
| RetinaNet | 70.18% | 65.57% | 67.80% | 71.70% | 48.45% |
| **Faster R-CNN** | **76.28%** | **85.66%** | **80.70%** | **82.81%** | **63.91%** |

### Best Performing Models

- **Best Classification Model:** ViT — **95.08% Accuracy** and **94.84% F1-score**
- **Best Detection Model:** Faster R-CNN — **82.81% mAP@0.5** and **63.91% mAP@0.5:0.95**
### YOLOv8m Per-Class Detection Performance

| Defect Class | mAP@0.5 |
|---|---:|
| Cut | 85.87% |
| Hole | 69.08% |
| Stain | 60.34% |
| ThreadError | 40.92% |

The YOLOv8m model achieved its highest per-class mAP@0.5 on **Cut (85.87%)**, while **ThreadError (40.92%)** was the most challenging defect class.
## Dataset

This project focuses on fabric defect detection and classification using a custom fabric defect dataset.

### Defect Classes

The dataset contains four major fabric defect categories:

1. **Cut**
2. **Hole**
3. **Stain**
4. **ThreadError**

The dataset was used for both object detection and image classification experiments with multiple deep learning architectures.

### Dataset Preparation

The dataset was processed and prepared for different model requirements, including:

- Object detection dataset preparation for YOLOv8, RetinaNet, and Faster R-CNN
- Classification dataset preparation for ViT, VGG16-BN, EfficientNet-B0, FDNet, and FDNet-V2
- Image preprocessing and resizing
- Training and validation data preparation
- Bounding-box based defect localization for detection models
## Models Used

This project evaluates multiple deep learning architectures for fabric defect classification and object detection.

### Classification Models

- **Vision Transformer (ViT)**
- **VGG16-BN**
- **EfficientNet-B0**
- **FDNet**
- **FDNet-V2**

### Object Detection Models

- **YOLOv8n**
- **YOLOv8m**
- **RetinaNet**
- **Faster R-CNN**

The models were trained and evaluated using the same fabric defect dataset, with model-specific preprocessing and training configurations.
## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/niamul-anto/Fabric-Defect-Detection.git
cd Fabric-Defect-Detection
