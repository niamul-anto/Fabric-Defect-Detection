# FD-Net V2: A Scratch-Trained Lightweight Deep Learning Architecture for Fabric Defect Diagnosis

A deep learning framework for fabric defect diagnosis using object detection and image classification. The repository evaluates several established deep learning models and contains the implementation of **FD-Net V2**, a lightweight image-classification architecture trained entirely from scratch without ImageNet or other external pretrained features.

## Dataset

The source dataset used in this study is the publicly available **Fabric defect detection** dataset hosted on **Roboflow Universe**.

**Canonical dataset source:**  
https://universe.roboflow.com/yolov7-vdg8u/fabric-defect-detection-jdyz3

The Roboflow Universe project contains **2,657 images** and four defect classes:

1. Cut
2. Hole
3. Stain
4. ThreadError

The source dataset is licensed under **CC BY 4.0**. The original dataset should be cited and attributed according to the information provided on the Roboflow Universe project page.

The authors **did not independently collect or create the source dataset**. For this study, the authors performed dataset verification, annotation processing, bounding-box validation, experimental splitting, CSV-to-YOLO conversion, defect-focused crop generation, model-specific preprocessing, training, and evaluation.

### Dataset citation

Roboflow provides the following citation information for the source dataset:

```bibtex
@misc{fabric-defect-detection-jdyz3_dataset,
  title        = {Fabric defect detection Dataset},
  type         = {Open Source Dataset},
  author       = {yolov7},
  howpublished = {\url{https://universe.roboflow.com/yolov7-vdg8u/fabric-defect-detection-jdyz3}},
  url          = {https://universe.roboflow.com/yolov7-vdg8u/fabric-defect-detection-jdyz3},
  journal      = {Roboflow Universe},
  publisher    = {Roboflow},
  year         = {2024},
  month        = {mar}
}
```

## Dataset Classes

| Class | Description |
|---|---|
| Cut | Cutting, tearing, or discontinuity in the fabric structure. |
| Hole | An opening or missing region in the fabric. |
| Stain | Localized discoloration or contamination on the fabric surface. |
| ThreadError | Thread-related structural defects such as broken, missing, displaced, or incorrectly woven threads. |

## Experimental Dataset Statistics

### Object Detection

The experimental split used in this study contains 2,657 images and 3,314 annotated defect instances.

| Split | Images | Annotation instances |
|---|---:|---:|
| Training | 2,000 | 2,401 |
| Validation | 500 | 669 |
| Test | 157 | 244 |
| **Total** | **2,657** | **3,314** |

### Image Classification

Defect-focused classification crops were generated from the annotated bounding boxes.

| Split | Cut | Hole | Stain | ThreadError | Total |
|---|---:|---:|---:|---:|---:|
| Training | 548 | 524 | 349 | 974 | 2,395 |
| Validation | 168 | 143 | 161 | 196 | 668 |
| Test | 41 | 62 | 51 | 90 | 244 |
| **Total** | **757** | **729** | **561** | **1,260** | **3,307** |

## Dataset Preparation

The public source dataset was adapted for two complementary experimental tasks: object detection and defect-focused image classification.

The preparation pipeline included:

1. Image and annotation organization
2. Dataset consistency checking
3. Bounding-box validation
4. Train/validation/test organization
5. CSV annotation processing
6. CSV-to-YOLO annotation conversion
7. Defect-focused classification crop generation
8. Model-specific preprocessing
9. Image resizing and training-time augmentation

The annotations used in the study pipeline were processed in CSV bounding-box form:

```text
filename,xmin,ymin,xmax,ymax,class
```

CSV annotations were converted to normalized YOLO annotations using:

```bash
python convert_to_yolo.py
```

Defect-focused classification crops were generated from bounding boxes using:

```bash
python create_classification_dataset.py
```

## Dataset Workflow

### Object Detection

```text
Public Roboflow Universe dataset
        ↓
Images + bounding-box annotations
        ↓
Annotation verification and validation
        ↓
Train / Validation / Test split
        ↓
Model-specific annotation preparation
        ↓
YOLOv8m / RetinaNet / Faster R-CNN
```

### Image Classification

```text
Annotated source image
        ↓
Bounding box
        ↓
Defect-focused crop
        ↓
224 × 224 preprocessing
        ↓
Class-specific dataset
        ↓
ViT-B/16 / VGG16-BN / EfficientNet-B0 / FD-Net V2
```

## Dataset Directory Examples

### YOLO Object-Detection Dataset

```text
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
```

### Classification Dataset

```text
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
```

## Image Preprocessing

Classification crops are resized to **224 × 224** pixels. Model-specific normalization and augmentation are applied during training. Validation and test samples are evaluated without random training augmentation.

## Models Reported in the Manuscript

### Object Detection

- YOLOv8m
- RetinaNet
- Faster R-CNN

### Image Classification

- ViT-B/16
- VGG16-BN
- EfficientNet-B0
- **FD-Net V2 (proposed; trained from scratch)**

### Additional Development Experiments

The repository may also contain scripts or outputs for **YOLOv8n** and the earlier **FD-Net** model. These were development/preliminary experiments and are **not part of the final comparative result tables reported in the manuscript**.

## Independent-Test Object Detection Performance

| Model | Precision (%) | Recall (%) | F1 (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) |
|---|---:|---:|---:|---:|---:|
| YOLOv8m | 75.78 | 50.00 | 60.25 | 64.04 | 39.06 |
| RetinaNet | 70.18 | 65.57 | 67.80 | 71.70 | 48.45 |
| **Faster R-CNN** | **76.28** | **85.66** | **80.70** | **82.81** | **63.91** |

### Best Detection Result

Under the final independent-test evaluation, **Faster R-CNN** achieved the strongest overall localization performance:

- Precision: **76.28%**
- Recall: **85.66%**
- F1-score: **80.70%**
- mAP@0.5: **82.81%**
- mAP@0.5:0.95: **63.91%**

These results describe performance under the evaluation protocol used in this study and should not be interpreted as a general claim that two-stage detectors are always superior.

## Independent-Test Image Classification Performance

The reported precision, recall, and F1 values use equal class contribution.

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---:|---:|---:|---:|
| **ViT-B/16** | **95.08** | **94.28** | **95.55** | **94.84** |
| VGG16-BN | 94.26 | 93.36 | 95.06 | 93.87 |
| EfficientNet-B0 | 92.62 | 91.91 | 93.37 | 92.22 |
| FD-Net V2 | 89.75 | 88.81 | 91.21 | 89.45 |

### FD-Net V2 Classification Result

FD-Net V2 achieved:

- Accuracy: **89.75%**
- Precision: **88.81%**
- Recall: **91.21%**
- F1-score: **89.45%**
- Trainable parameters: **3.07 million**

FD-Net V2 was randomly initialized and trained entirely on the study dataset. Unlike the pretrained comparison models, it did not use ImageNet or another external pretrained feature extractor.

Although ViT-B/16 achieved the highest classification accuracy, FD-Net V2 provides a substantially more compact model and is evaluated primarily in terms of its **accuracy-resource trade-off**, rather than as the highest-accuracy classifier.

## FD-Net V2 Computational Benchmark

The four manuscript-reported classification models were benchmarked using a common inference configuration:

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- Inference batch size: 1 (benchmarking only; not the training batch size)
- Inference precision: FP32
- Input size: 224 × 224
- CUDA synchronization around timing events
- 100 warm-up runs
- 5 repetitions of 200 forward passes

| Model | Params (M) ↓ | Size (MB) ↓ | Est. GFLOPs ↓ | Peak GPU (MB) ↓ | Latency (ms/image) ↓ | Throughput (images/s) ↑ |
|---|---:|---:|---:|---:|---:|---:|
| ViT-B/16 | 85.80 | 327.37 | 16.87 | 344.08 | 3.073 | 325.45 |
| VGG16-BN | 134.29 | 512.32 | 15.49 | 2098.24 | **2.345** | **426.36** |
| EfficientNet-B0 | 4.01 | 15.59 | **0.400** | 80.53 | 3.683 | 271.55 |
| **FD-Net V2** | **3.07** | **11.88** | 0.899 | **63.38** | 3.011 | 332.09 |

### Efficiency Interpretation

FD-Net V2 has:

- The lowest parameter count: **3.07M**
- The smallest serialized model footprint: **11.88 MB**
- The lowest measured peak GPU memory: **63.38 MB**
- FP32 batch-1 latency: **3.011 ms/image**
- Throughput: **332.09 images/s**

EfficientNet-B0 has the lowest estimated GFLOPs, while VGG16-BN has the lowest measured GPU latency in the stated benchmark. Therefore, FD-Net V2 is **not claimed to be the best model on every efficiency metric**. Its principal advantages are compact parameter count, small storage requirement, low measured GPU memory use, and competitive classification performance without external pretraining.

## FD-Net V2 Architecture

FD-Net V2 is a lightweight classifier developed for fabric-defect classification. Its design combines:

- Inverted residual learning
- Depthwise convolution
- Residual shortcuts
- Selective channel-spatial attention
- Global average pooling
- A compact final classifier

The final architecture contains approximately **3.07 million trainable parameters** and is trained entirely from scratch.

## Training and Independent Evaluation

Training, validation, and testing are kept separate:

- Training data are used for parameter optimization.
- Validation data are used for learning-rate adjustment, early stopping, and checkpoint selection.
- The independent test set is used only after model selection.
- Classification checkpoints are selected by validation F1-score.

FD-Net V2 uses AdamW with:

- Initial learning rate: **3 × 10^-4**
- Weight decay: **1 × 10^-4**
- Scheduler: cosine annealing
- Early-stopping patience: **20 epochs**

Where CUDA mixed precision is used during training, the final computational benchmark values reported above are measured in FP32 for a consistent comparison.

## Training Scripts

### Main manuscript experiments

YOLOv8m:

```bash
python train_yolov8m.py
```

RetinaNet:

```bash
python train_retinanet.py
```

Faster R-CNN:

```bash
python train_fasterrcnn.py
```

VGG16-BN:

```bash
python train_vgg16_classification.py
```

EfficientNet-B0:

```bash
python train_efficientnet_b0_final.py
```

Vision Transformer:

```bash
python train_vit_b16_final.py
```

FD-Net V2:

```bash
python train_fdnet_v2.py
```

### Additional development scripts

If retained in the repository, scripts for YOLOv8n and the earlier FD-Net model should be treated as development/preliminary experiments rather than part of the final manuscript comparison.

## Evaluation and Benchmark Scripts

Detection:

```bash
python evaluate_yolov8m.py
python evaluate_retinanet.py
python evaluate_fasterrcnn.py
```

YOLOv8m threshold analysis:

```bash
python evaluate_yolov8m_thresholds.py
```

Classification/computational benchmarks:

```bash
python benchmark_efficientnet_b0.py
python benchmark_vgg16_bn.py
python benchmark_vit_b16.py
python benchmark_fdnet_v2.py
```

## Reproducibility

Install the project dependencies using:

```bash
pip install -r requirements.txt
```

For a publication/release snapshot, an exact environment lock file is recommended. Generate it from the environment actually used for the reported experiments:

```bash
pip freeze > requirements-lock.txt
```

Do **not** replace the lock file with guessed package versions. The lock file should reflect the actual experimental environment.

Random seeds should be documented in the relevant training scripts. Exact numerical reproducibility across different CUDA, cuDNN, PyTorch, driver, and hardware versions is not guaranteed unless deterministic execution is explicitly configured.

## Research Contributions

The main contributions reported in the manuscript are:

1. Proposal of **FD-Net V2**, a lightweight scratch-trained architecture for fabric-defect classification.
2. A unified study of complementary fabric-defect localization and classification tasks using a common four-class experimental setting.
3. Comparative evaluation of **YOLOv8m, RetinaNet, and Faster R-CNN** for defect localization.
4. Comparative evaluation of **ViT-B/16, VGG16-BN, EfficientNet-B0, and FD-Net V2** for defect-focused classification.
5. Dataset-processing workflows for bounding-box verification, CSV-to-YOLO conversion, and defect-focused crop generation.
6. Computational benchmarking of the manuscript-reported classifiers using parameter count, model size, estimated GFLOPs, FP32 latency, throughput, and peak GPU memory.
7. Industrial review of representative defect samples, categories, labels, and annotations.

## Industrial Review

Representative dataset samples, defect categories, class labels, and annotations were reviewed from an industrial garment-manufacturing perspective by professionals from:

**JOYTEX SOURCING LTD., Dhaka, Bangladesh**

The review was used to assess the industrial relevance of representative defect categories and annotations. **JOYTEX SOURCING LTD. did not create, own, or provide the public source dataset used in the experiments.**

## Limitations

The present study has several limitations:

- Only four defect categories are considered.
- The source dataset does not represent every possible fabric type, weave structure, color, defect severity, or production environment.
- External cross-factory validation has not yet been performed.
- The experiments are based primarily on static images.
- Continuous production-line video data are not included.
- Classification is performed on defect-focused crops rather than as a fully integrated end-to-end production pipeline.
- Benchmarking was performed on an NVIDIA GeForce RTX 4070 Ti SUPER; power consumption, energy use, and embedded/edge-device performance were not measured.
- Real-world deployment may introduce changes in illumination, camera position, fabric texture, and background conditions.

## Future Work

Future work may investigate:

- Larger and more diverse fabric-defect datasets
- Additional defect categories
- Cross-dataset and multi-factory validation
- Multiple random-seed experiments
- Different fabric types and textures
- Variable illumination and camera conditions
- Defect segmentation
- Video-based production-line inspection
- Edge-device deployment
- Quantization, pruning, and knowledge distillation
- Power and energy benchmarking on embedded hardware

## Data Availability

The source dataset is publicly available from Roboflow Universe:

https://universe.roboflow.com/yolov7-vdg8u/fabric-defect-detection-jdyz3

The source dataset is subject to its original **CC BY 4.0** license and attribution requirements.

This repository contains the processing and experimental code used for the study. Source or derived dataset material should be redistributed only in accordance with the original dataset license and applicable terms.

## Code Availability

The code used for data preparation, model training, evaluation, and computational benchmarking, including the implementation of FD-Net V2, is available in this repository:

https://github.com/niamul-anto/Fabric-Defect-Detection

For publication-quality reproducibility, releases should identify the exact commit/tag corresponding to the manuscript version.

## Citation

If you use FD-Net V2, this repository, the experimental pipeline, or the reported results in academic work, please cite the associated research paper once final publication details are available.

Repository citation metadata is provided in:

```text
CITATION.cff
```

The **original Roboflow Universe dataset must also be cited separately**.

## Acknowledgments

The authors acknowledge the creators and contributors of the public **Fabric defect detection** dataset hosted on Roboflow Universe.

The authors also acknowledge **JOYTEX SOURCING LTD., Dhaka, Bangladesh** for industrial review of representative fabric-defect samples, categories, labels, and annotations.

JOYTEX SOURCING LTD. was not the source of the public dataset.

## License

The public source dataset is licensed separately under **CC BY 4.0** by its source provider.

The repository code and author-created research materials are distributed according to the license specified in the repository's `LICENSE` file. The dataset license and the code license are separate and should not be conflated.

## Contact

**Md. Niamul Islam Khan**  
Department of Computer Science and Engineering  
BRAC University  
Dhaka, Bangladesh

**Md. Al Mamunur Rashid Emon**  
Department of Computer Science and Engineering  
BRAC University  
Dhaka, Bangladesh

**Miskatunnisa Labonno**  
Department of Computer Science and Engineering  
BRAC University  
Dhaka, Bangladesh

**Mohoshin Al Mamun**  
Department of Computer Science and Engineering  
BRAC University  
Dhaka, Bangladesh
