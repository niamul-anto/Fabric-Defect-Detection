# FD-Net V2: A Scratch-Trained Lightweight Deep Learning Architecture for Fabric Defect Diagnosis

A deep learning-based framework for fabric defect detection and diagnosis using object detection and image classification. This project evaluates multiple deep learning architectures and proposes FD-Net V2, a lightweight classification architecture trained entirely from scratch without external pretrained features.

## Dataset

The dataset used in this study was obtained from the publicly available Fabric Defect Detection dataset hosted on Roboflow Universe.

Dataset Source:
https://universe.roboflow.com/niamul-khan-anto/fabric-defect-detection-jdyz3-bvivw

The dataset contains four fabric defect categories:

1. Cut
2. Hole
3. Stain
4. ThreadError

The experimental dataset contains 2,657 images, 3,314 annotated defect instances, and 4 defect classes.

The original annotations were provided in CSV format using the following structure:

filename,xmin,ymin,xmax,ymax,class

The source dataset is used with appropriate attribution according to its applicable license.

## Dataset Classes

| Class | Description |
|---|---|
| Cut | Cutting, tearing, or discontinuity in the fabric structure. |
| Hole | An opening or missing region in the fabric. |
| Stain | Localized discoloration or contamination on the fabric surface. |
| ThreadError | Thread-related structural defects such as broken, missing, displaced, or incorrectly woven threads. |

## Dataset Statistics

### Object Detection Dataset

| Split | Images | Annotation Rows |
|---|---:|---:|
| Training | 2,000 | 2,401 |
| Validation | 500 | 669 |
| Test | 157 | 244 |
| Total | 2,657 | 3,314 |

### Classification Dataset

Defect-focused crops were generated from the annotated bounding boxes.

| Split | Cut | Hole | Stain | ThreadError | Total |
|---|---:|---:|---:|---:|---:|
| Training | 548 | 524 | 349 | 974 | 2,395 |
| Validation | 168 | 143 | 161 | 196 | 668 |
| Test | 41 | 62 | 51 | 90 | 244 |
| Total | 757 | 729 | 561 | 1,260 | 3,307 |

## Dataset Preparation

The source dataset was adapted for both object-detection and image-classification experiments.

The preparation process included:

1. Image and annotation organization
2. Dataset consistency checking
3. Bounding-box validation
4. Train/validation/test organization
5. CSV annotation processing
6. CSV-to-YOLO annotation conversion
7. Defect-focused classification crop generation
8. Model-specific preprocessing
9. Image resizing and augmentation

The source annotations were available in CSV format rather than YOLO TXT format.

The CSV annotations were converted into normalized YOLO annotations using convert_to_yolo.py.

Classification crops were generated from the bounding boxes using create_classification_dataset.py.

## Dataset Workflow

Roboflow Universe Dataset
→ Images + CSV Annotations
→ Annotation Validation
→ Train / Validation / Test
→ CSV-to-YOLO Conversion
→ Object Detection

For classification:

Original Image
→ Bounding Box
→ Defect Crop
→ Class-specific Folder
→ Classification Models

## YOLO Dataset Structure

yolo_dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

Run the conversion with:

python convert_to_yolo.py

## Classification Dataset Structure

classification_dataset/
├── train/
│   ├── Cut/
│   ├── Hole/
│   ├── Stain/
│   └── ThreadError/
├── valid/
│   ├── Cut/
│   ├── Hole/
│   ├── Stain/
│   └── ThreadError/
└── test/
    ├── Cut/
    ├── Hole/
    ├── Stain/
    └── ThreadError/

Run:

python create_classification_dataset.py

## Image Preprocessing

Classification images are resized to 224 × 224 pixels.

Model-specific normalization and augmentation are applied during training. Validation and test samples are evaluated without random training augmentation.

## Models Evaluated

### Object Detection

- YOLOv8n
- YOLOv8m
- RetinaNet
- Faster R-CNN

### Image Classification

- ViT-B/16
- VGG16-BN
- EfficientNet-B0
- FD-Net
- FD-Net V2

## Classification Performance

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 92.62% | 91.91% | 93.37% | 92.22% |
| FD-Net | 62.70% | 62.25% | 61.55% | 60.54% |
| FD-Net V2 | 89.75% | 88.81% | 91.21% | 89.45% |
| VGG16-BN | 94.26% | 93.36% | 95.06% | 93.87% |
| ViT-B/16 | 95.08% | 94.28% | 95.55% | 94.84% |

## FD-Net V2 Results

FD-Net V2 achieved:

- 89.75% Accuracy
- 88.81% Precision
- 91.21% Recall
- 89.45% F1-score
- 3.07M trainable parameters

FD-Net V2 was trained entirely from scratch without external pretrained features.

Although ViT-B/16 achieved higher classification accuracy, FD-Net V2 provides a substantially more compact model with a favorable accuracy-resource trade-off.

## Object Detection Performance

| Model | Precision | Recall | F1-score | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 56.60% | 60.83% | — | 59.85% | 32.54% |
| YOLOv8m | 72.39% | 60.65% | — | 64.05% | 39.10% |
| RetinaNet | 71.70% | 58.93% | 64.69% | 71.70% | 48.45% |
| Faster R-CNN | 82.81% | 72.50% | 77.31% | 82.81% | 63.91% |

## Best Detection Result

Faster R-CNN achieved:

- 82.81% mAP@0.5
- 63.91% mAP@0.5:0.95
- 82.81% Precision
- 72.50% Recall
- 77.31% F1-score

## YOLOv8m Per-Class Performance

| Defect Class | mAP@0.5 |
|---|---:|
| Cut | 85.87% |
| Hole | 69.08% |
| Stain | 60.34% |
| ThreadError | 40.92% |

YOLOv8m achieved its highest per-class mAP@0.5 on Cut, while ThreadError was the most challenging category.

## FD-Net V2 Computational Benchmark

The classification models were benchmarked under a common inference configuration:

- NVIDIA GeForce RTX 4070 Ti SUPER
- Batch size: 1
- FP32 precision
- Input size: 224 × 224
- CUDA synchronization during timing
- 100 warm-up runs
- 5 repetitions of 200 forward passes

| Model | Params (M) ↓ | Size (MB) ↓ | Est. GFLOPs ↓ | Peak GPU (MB) ↓ | Latency (ms/image) ↓ | Throughput (images/s) ↑ |
|---|---:|---:|---:|---:|---:|---:|
| ViT-B/16 | 85.80 | 327.37 | 16.87 | 344.08 | 3.073 | 325.45 |
| VGG16-BN | 134.29 | 512.32 | 15.49 | 2098.24 | 2.345 | 426.36 |
| EfficientNet-B0 | 4.01 | 15.59 | 0.400 | 80.53 | 3.683 | 271.55 |
| FD-Net V2 | 3.07 | 11.88 | 0.899 | 63.38 | 3.011 | 332.09 |

## FD-Net V2 Efficiency Highlights

- Lowest parameter count: 3.07M
- Smallest model footprint: 11.88 MB
- Lowest peak GPU memory: 63.38 MB
- Latency: 3.011 ms/image
- Throughput: 332.09 images/s

FD-Net V2 therefore provides a favorable balance between classification performance and computational resource requirements.

## FD-Net V2 Architecture

FD-Net V2 is the proposed lightweight architecture developed specifically for fabric-defect classification.

The architecture incorporates:

- Inverted residual learning
- Depthwise convolution
- Selective channel-spatial attention
- Residual connections
- Global pooling

Unlike the pretrained comparison models, FD-Net V2 is trained entirely from scratch.

## Training and Evaluation

Training, validation, and testing are kept separate.

- Training data are used for parameter optimization.
- Validation data are used for learning-rate adjustment, early stopping, and checkpoint selection.
- The independent test set is used only after model selection.

Classification checkpoints are selected using validation F1-score.

FD-Net V2 uses AdamW with:

- Initial learning rate: 3 × 10^-4
- Weight decay: 1 × 10^-4
- Scheduler: Cosine annealing
- Early stopping patience: 20 epochs

## Training Scripts

YOLOv8n:
python train_yolov8.py

YOLOv8m:
python train_yolov8m.py

RetinaNet:
python train_retinanet.py

Faster R-CNN:
python train_fasterrcnn.py

VGG16-BN:
python train_vgg16_classification.py

EfficientNet-B0:
python train_efficientnet_b0_final.py

Vision Transformer:
python train_vit_b16_final.py

FD-Net:
python train_fdnet_final.py

FD-Net V2:
python train_fdnet_v2.py

## Evaluation Scripts

Detection:

python evaluate_yolov8m.py
python evaluate_retinanet.py
python evaluate_fasterrcnn.py

YOLOv8m threshold analysis:

python evaluate_yolov8m_thresholds.py

Classification benchmarks:

python benchmark_efficientnet_b0.py
python benchmark_vgg16_bn.py
python benchmark_vit_b16.py
python benchmark_fdnet_v2.py

## Research Contributions

The main contributions of this work are:

1. Proposal of FD-Net V2, a lightweight scratch-trained deep learning architecture for fabric defect diagnosis.
2. Systematic evaluation of multiple deep learning architectures for fabric-defect detection and classification.
3. A CSV-based bounding-box processing pipeline.
4. A custom CSV-to-YOLO annotation conversion pipeline.
5. An automated bounding-box-based classification crop generation pipeline.
6. Comparative evaluation of YOLOv8n, YOLOv8m, RetinaNet, and Faster R-CNN for defect localization.
7. Comparative evaluation of ViT-B/16, VGG16-BN, EfficientNet-B0, FD-Net, and FD-Net V2 for defect classification.
8. Computational benchmarking based on parameters, model size, GFLOPs, latency, throughput, and peak GPU memory.
9. Industrial review of representative defect categories and annotations.

## Industrial Review

Representative dataset samples, defect categories, labels, and annotations were reviewed from an industrial garment-manufacturing perspective by professionals from:

JOYTEX SOURCING LTD., Dhaka, Bangladesh

The review assessed the industrial relevance of representative fabric-defect categories and annotations.

JOYTEX SOURCING LTD. served as an industrial review source and did not create or provide the dataset used in the experiments.

## Limitations

The current study has several limitations:

- Only four defect categories are considered.
- The dataset does not represent every possible fabric type, weave structure, color, defect severity, or production environment.
- External cross-factory validation has not yet been performed.
- The experiments are based primarily on static images.
- Continuous production-line video data are not included.
- Additional defect categories may require further model adaptation.
- Real-world deployment may introduce changes in illumination, camera position, fabric texture, and background conditions.

## Future Work

Future work may investigate:

- Larger and more diverse fabric-defect datasets
- Additional defect categories
- Multi-factory validation
- Different fabric types and textures
- Different camera systems
- Variable illumination conditions
- Real-time production-line inspection
- Video-based defect detection
- Edge-device deployment
- Model compression and quantization
- External dataset validation

## Dataset Availability

The dataset used in this project was obtained from the publicly available Fabric Defect Detection dataset hosted on Roboflow Universe.

Source:
https://universe.roboflow.com/niamul-khan-anto/fabric-defect-detection-jdyz3-bvivw

This repository provides the processing and experimental pipelines used to adapt the source dataset, including:

- CSV annotation processing
- Bounding-box validation
- CSV-to-YOLO conversion
- Classification crop generation
- Model-specific preprocessing

Any redistribution or reuse of source or derived dataset material must comply with the applicable license and attribution requirements of the original dataset.

## Citation

If you use this repository, FD-Net V2, the experimental pipeline, or the reported results in academic work, please cite the associated research paper once the final publication information becomes available.

Repository citation metadata is provided through:

CITATION.cff

The original dataset should also be appropriately attributed to its source on Roboflow Universe.

## Acknowledgments

We acknowledge the original creators and contributors of the Fabric Defect Detection dataset hosted on Roboflow Universe.

We also acknowledge JOYTEX SOURCING LTD., Dhaka, Bangladesh, for their industrial review of representative fabric-defect samples, categories, labels, and annotations.

## License

The source dataset is subject to its original licensing and attribution requirements.

The code and research materials developed in this repository are distributed according to the license specified in:

LICENSE

Users must comply with the applicable license and attribution requirements when using or redistributing source or derived dataset material.

## Contact

Md. Niamul Islam Khan
Department of Computer Science and Engineering
Brac University
Dhaka, Bangladesh

Md. Al Mamunur Rashid Emon
Department of Computer Science and Engineering
Brac University
Dhaka, Bangladesh

Miskatunnisa Labonno
Department of Computer Science and Engineering
Brac University
Dhaka, Bangladesh

Rubiya Tasfi Bidisha
Department of Computer Science and Engineering
Brac University
Dhaka, Bangladesh