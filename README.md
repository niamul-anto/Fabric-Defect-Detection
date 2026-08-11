# Fabric Defect Detection

A deep learning-based fabric defect detection and classification project using multiple computer vision models.

## 📌 Project Overview

This project focuses on detecting and classifying defects in fabric images using deep learning and computer vision techniques.

Multiple object detection and image classification models were trained and evaluated to compare their performance on the fabric defect dataset.

## 🎯 Defect Classes

The dataset contains four major fabric defect classes:

1. Sewing Defects
2. Cotton Defects
3. Hole
4. Color Defects

## 🤖 Models Used

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

## 📂 Project Structure

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