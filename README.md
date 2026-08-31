# FD-Net V2: A Scratch-Trained Lightweight Deep Learning Architecture for Fabric Defect Diagnosis

A deep-learning framework for four-class fabric defect diagnosis using complementary object-detection and defect-focused image-classification pipelines.

This repository contains the implementation and experiments for **FD-Net V2**, a lightweight image-classification architecture that is randomly initialized and trained from scratch without ImageNet or another external pretrained feature extractor. It also includes **FD-Net V2-KD**, a knowledge-distilled extension in which a frozen ImageNet-pretrained ViT-B/16 teacher transfers soft-target information to a randomly initialized FD-Net V2 student during training. The teacher is not required at inference.

> **Important terminology**
>
> - **FD-Net V2**: scratch-trained baseline.
> - **FD-Net V2-KD**: randomly initialized FD-Net V2 student trained with knowledge distillation from an ImageNet-pretrained ViT-B/16 teacher.
>
> Therefore, the KD variant should not be described as being trained without external pretrained knowledge.

## Dataset

The source dataset used in this study is the publicly available **Fabric defect detection** dataset hosted on **Roboflow Universe**.

**Canonical dataset source:**  
https://universe.roboflow.com/yolov7-vdg8u/fabric-defect-detection-jdyz3

The public project contains **2,657 images** and four defect classes:

1. Cut
2. Hole
3. Stain
4. ThreadError

The source dataset is licensed under **CC BY 4.0**.

The authors **did not independently collect or create the source dataset**. For this study, the authors performed dataset verification, annotation processing, bounding-box validation, experimental splitting, CSV-to-YOLO conversion, defect-focused crop generation, model-specific preprocessing, training, evaluation, and error analysis.

Representative samples, defect categories, labels, and annotations were also reviewed from an industrial garment-manufacturing perspective by professionals from **JOYTEX SOURCING LTD., Dhaka, Bangladesh**. JOYTEX SOURCING LTD. did **not** create, own, or provide the public source dataset.

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

## Dataset classes

| Class | Description |
|---|---|
| Cut | Cutting, tearing, or discontinuity in the fabric structure |
| Hole | An opening or missing region in the fabric |
| Stain | Localized discoloration or contamination on the fabric surface |
| ThreadError | Thread-related structural defects such as broken, missing, displaced, or incorrectly woven threads |

## Experimental dataset statistics

### Object detection

The experimental split contains 2,657 images and 3,314 annotated defect instances.

| Split | Images | Annotation instances |
|---|---:|---:|
| Training | 2,000 | 2,401 |
| Validation | 500 | 669 |
| Test | 157 | 244 |
| **Total** | **2,657** | **3,314** |

### Image classification

Defect-focused classification crops were generated from the annotated bounding boxes.

| Split | Cut | Hole | Stain | ThreadError | Total |
|---|---:|---:|---:|---:|---:|
| Training | 548 | 524 | 349 | 974 | 2,395 |
| Validation | 168 | 143 | 161 | 196 | 668 |
| Test | 41 | 62 | 51 | 90 | 244 |
| **Total** | **757** | **729** | **561** | **1,260** | **3,307** |

## Dataset preparation

The public source dataset was adapted for two complementary tasks: object detection and defect-focused image classification.

The preparation workflow includes:

1. Image and annotation organization
2. Dataset consistency checking
3. Bounding-box verification and validation
4. Train/validation/test organization
5. CSV annotation processing
6. CSV-to-YOLO annotation conversion
7. Defect-focused classification crop generation
8. Model-specific preprocessing
9. Image resizing and training-time augmentation

CSV bounding-box format:

```text
filename,xmin,ymin,xmax,ymax,class
```

CSV-to-YOLO conversion:

```bash
python convert_to_yolo.py
```

Classification crop generation:

```bash
python create_classification_dataset.py
```

## Experimental workflow

### Object detection

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
        ↓
Independent test evaluation
```

### Image classification and knowledge distillation

```text
Annotated source images
        ↓
Bounding-box defect crops
        ↓
224 × 224 preprocessing
        ↓
ViT-B/16 / VGG16-BN / EfficientNet-B0 / FD-Net V2
        ↓
Validation-based checkpoint selection
        ↓
Independent test evaluation

ViT-B/16 teacher (frozen)
        ↓ soft targets
FD-Net V2 student (random initialization)
        ↓
Knowledge-distillation training
        ↓
Validation-only KD hyperparameter selection
        ↓
Independent test evaluation of selected FD-Net V2-KD
```

## Dataset directory examples

### YOLO object-detection dataset

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

### Classification dataset

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

## Image preprocessing

Classification crops are resized to **224 × 224** pixels.

The FD-Net V2 / KD training pipeline uses:

- `Resize((256, 256))`
- `RandomResizedCrop(224, scale=(0.85, 1.00), ratio=(0.90, 1.10))`
- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(degrees=7)`
- `ColorJitter(brightness=0.12, contrast=0.15, saturation=0.08, hue=0.01)`
- `ToTensor()`
- ImageNet mean/std normalization

Validation and test preprocessing use deterministic resizing to 224 × 224 followed by tensor conversion and ImageNet normalization.

## Models

### Object detection

- YOLOv8m
- RetinaNet
- Faster R-CNN

### Image classification

- ViT-B/16
- VGG16-BN
- EfficientNet-B0
- **FD-Net V2** — proposed scratch-trained lightweight classifier
- **FD-Net V2-KD** — knowledge-distilled FD-Net V2 student

### Additional development experiments

The repository may also contain scripts or outputs for YOLOv8n and an earlier FD-Net model. These are development/preliminary experiments and are not part of the final manuscript comparison.

## Independent-test object-detection performance

| Model | Precision (%) | Recall (%) | F1 (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) |
|---|---:|---:|---:|---:|---:|
| YOLOv8m | 75.78 | 50.00 | 60.25 | 64.04 | 39.06 |
| RetinaNet | 70.18 | 65.57 | 67.80 | 71.70 | 48.45 |
| **Faster R-CNN** | **76.28** | **85.66** | **80.70** | **82.81** | **63.91** |

Under the final independent-test evaluation, Faster R-CNN achieved the strongest overall localization performance among the evaluated detectors. This result applies to the dataset and protocol used here and is not a general claim that two-stage detectors are always superior.

## Independent-test image-classification performance

Precision, recall, and F1 use equal class contribution (macro averaging).

| Model | Accuracy (%) | Precision (%) | Recall (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|
| **ViT-B/16** | **95.08** | **94.28** | **95.55** | **94.84** |
| VGG16-BN | 94.26 | 93.36 | 95.06 | 93.87 |
| EfficientNet-B0 | 92.62 | 91.91 | 93.37 | 92.22 |
| FD-Net V2 | 89.75 | 88.81 | 91.21 | 89.45 |
| **FD-Net V2-KD** | **92.21** | **91.25** | **92.76** | **91.73** |

Knowledge distillation improved FD-Net V2 test accuracy from **89.75% to 92.21% (+2.46 percentage points)** and macro-F1 from **89.45% to 91.73% (+2.28 percentage points)**.

ViT-B/16 remained the highest-accuracy classifier in the independent test evaluation.

## FD-Net V2 architecture

FD-Net V2 combines:

- Inverted residual learning
- Depthwise convolution
- Residual shortcuts
- Selective channel-spatial attention
- Global average pooling
- A compact final classifier

The architecture contains exactly **3,068,448 trainable parameters**.

The base FD-Net V2 is randomly initialized and trained from scratch.

## FD-Net V2 training

Training, validation, and testing are kept separate:

- Training data are used for parameter optimization.
- Validation data are used for learning-rate adjustment, early stopping, and checkpoint selection.
- The independent test set is used only after model selection.
- Classification checkpoints are selected using validation macro-F1.

FD-Net V2 uses AdamW with:

- Initial learning rate: **3 × 10^-4**
- Weight decay: **1 × 10^-4**
- Scheduler: cosine annealing
- Early-stopping patience: **20 epochs**
- Training batch size: **16**

## FD-Net V2-KD

### Distillation setup

- **Teacher:** ImageNet-pretrained ViT-B/16, fine-tuned on the training split and frozen during distillation
- **Student:** FD-Net V2, randomly initialized
- **Student training batch size:** 16
- **Optimizer:** AdamW
- **Initial learning rate:** 3 × 10^-4
- **Weight decay:** 1 × 10^-4
- **Scheduler:** cosine annealing
- **Early-stopping patience:** 20 epochs
- **Checkpoint selection:** validation macro-F1

The distillation objective is

```text
L = α L_CE + (1 - α) T² L_KD
```

where:

- `L_CE` is the ground-truth cross-entropy loss
- `L_KD` is the soft-target KL-divergence loss
- `T` is the distillation temperature
- `α` is the cross-entropy weight

### Validation-only hyperparameter search

The independent test set was not used to select the KD configuration.

| Configuration | Temperature T | Alpha α | Best validation macro-F1 (%) |
|---|---:|---:|---:|
| T4_A05 | 4 | 0.5 | 97.08 |
| T2_A05 | 2 | 0.5 | 97.18 |
| **T4_A07** | **4** | **0.7** | **98.04** |
| T6_A05 | 6 | 0.5 | 97.87 |

The selected configuration was **T = 4, α = 0.7**.

### Final independent-test result

The selected FD-Net V2-KD checkpoint was evaluated on the independent test set of 244 classification crops:

- Accuracy: **92.21%**
- Macro precision: **91.25%**
- Macro recall: **92.76%**
- Macro-F1: **91.73%**

The validation macro-F1 of 98.04% should not be compared directly with test metrics from other models because it was measured on a different split.

## Computational benchmarking

### Original four-classifier benchmark

The manuscript-reported classifier benchmark used:

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- Inference batch size: 1
- Precision: FP32
- Input size: 224 × 224
- CUDA synchronization around timing events
- 100 warm-up runs
- 5 repetitions × 200 forward passes

| Model | Params (M) | Size (MB) | Est. GFLOPs | Peak GPU (MB) | Latency (ms/image) | Throughput (images/s) |
|---|---:|---:|---:|---:|---:|---:|
| ViT-B/16 | 85.80 | 327.37 | 16.87 | 344.08 | 3.073 | 325.45 |
| VGG16-BN | 134.29 | 512.32 | 15.49 | 2098.24 | 2.345 | 426.36 |
| EfficientNet-B0 | 4.01 | 15.59 | 0.400 | 80.53 | 3.683 | 271.55 |
| FD-Net V2 | 3.07 | 11.88 | 0.899 | 63.38 | 3.011 | 332.09 |

This table reflects the original four-classifier benchmarking implementation. It should not be used to infer that knowledge distillation itself changes inference speed.

### Fair same-graph benchmark: FD-Net V2 vs FD-Net V2-KD

For the KD study, the baseline and distilled student were re-benchmarked using the **same canonical FD-Net V2 forward graph** and the same FP32 batch-1 protocol.

Legacy-to-canonical conversion of the baseline checkpoint was numerically verified with **maximum absolute output difference = 0**.

| Metric | FD-Net V2 | FD-Net V2-KD |
|---|---:|---:|
| Parameters | 3,068,448 | 3,068,448 |
| Serialized model size (MB) | 11.8732 | 11.8732 |
| Estimated GFLOPs | 0.898927 | 0.898927 |
| Peak GPU memory (MB) | 27.4312 | 27.4312 |

Knowledge distillation does **not** change the inference architecture. Accordingly, parameter count, serialized size, estimated GFLOPs, and measured peak GPU memory were identical in the fair same-graph comparison.

### Latency / throughput robustness check

Two benchmark orders were evaluated:

| Run order | Model | Latency (ms/image) | Throughput (images/s) |
|---|---|---:|---:|
| Baseline first | FD-Net V2 | 2.5085 | 398.64 |
| Baseline first | FD-Net V2-KD | 2.4899 | 401.63 |
| KD first | FD-Net V2 | 2.5087 | 398.61 |
| KD first | FD-Net V2-KD | 4.1744 | 239.56 |

The KD latency/throughput measurement showed run-order sensitivity. Therefore, this repository **does not claim that knowledge distillation caused an inference-speed improvement**. Latency and throughput differences are treated as runtime measurement variation unless stable across repeated benchmarking conditions.

## Main training scripts

### Object detection

```bash
python train_yolov8m.py
python train_retinanet.py
python train_fasterrcnn.py
```

### Classification

```bash
python train_vgg16_classification.py
python train_efficientnet_b0_final.py
python train_vit_b16_final.py
python train_fdnet_v2.py
```

### FD-Net V2 knowledge distillation

Core KD model:

```bash
fdnet_v2_kd.py
```

KD training / testing:

```bash
python train_fdnet_v2_kd.py train --help
python train_fdnet_v2_kd.py test --help
```

Selected experiment example:

```bash
python train_fdnet_v2_kd.py train \
  --data-root classification_dataset \
  --teacher-checkpoint results/vit_classification/best_vit.pt \
  --output-dir results/fdnet_v2_kd_T4_A07 \
  --temperature 4 \
  --alpha 0.7 \
  --batch-size 16
```

Final independent-test evaluation:

```bash
python train_fdnet_v2_kd.py test \
  --data-root classification_dataset \
  --student-checkpoint results/fdnet_v2_kd_T4_A07/fdnet_v2_kd_best.pth \
  --output-dir results/fdnet_v2_kd_final_test
```

## Evaluation and benchmark scripts

### Detection

```bash
python evaluate_yolov8m.py
python evaluate_retinanet.py
python evaluate_fasterrcnn.py
python evaluate_yolov8m_thresholds.py
```

### Classification / computational benchmark

```bash
python benchmark_efficientnet_b0.py
python benchmark_vgg16_bn.py
python benchmark_vit_b16.py
python benchmark_fdnet_v2.py
```

### KD benchmarking

```bash
python benchmark_fdnet_v2_kd.py
python benchmark_fdnet_v2_vs_kd_fair.py --order baseline-first
python benchmark_fdnet_v2_vs_kd_fair.py --order kd-first
```

## KD result directories

The following experiment outputs support the KD analysis:

```text
results/
├── fdnet_v2_kd_T4_A05/
├── fdnet_v2_kd_T2_A05/
├── fdnet_v2_kd_T4_A07/
├── fdnet_v2_kd_T6_A05/
├── fdnet_v2_kd_final_test/
├── fdnet_v2_kd_benchmark/
└── fdnet_v2_vs_kd_fair_benchmark/
```

Large model checkpoints may be excluded from GitHub and distributed separately where appropriate. CSV, JSON, TXT, figures, and other lightweight reproducibility outputs should be retained when possible.

## Reproducibility

Install project dependencies using:

```bash
pip install -r requirements.txt
```

For a publication/release snapshot, generate an exact environment lock file from the environment actually used:

```bash
pip freeze > requirements-lock.txt
```

Do not replace the lock file with guessed package versions.

Random seeds should be documented in the relevant training scripts. Exact numerical reproducibility across different CUDA, cuDNN, PyTorch, driver, and hardware versions is not guaranteed unless deterministic execution is explicitly configured.

## Research contributions

1. Proposal of **FD-Net V2**, a lightweight scratch-trained architecture for fabric-defect classification.
2. Unified evaluation of complementary fabric-defect localization and classification tasks under a common four-class setting.
3. Comparative evaluation of YOLOv8m, RetinaNet, and Faster R-CNN for defect localization.
4. Comparative evaluation of ViT-B/16, VGG16-BN, EfficientNet-B0, and FD-Net V2 for defect-focused classification.
5. Knowledge-distillation extension **FD-Net V2-KD**, using a frozen ViT-B/16 teacher and randomly initialized FD-Net V2 student.
6. Validation-only KD hyperparameter selection followed by a single selected independent-test evaluation.
7. Dataset-processing workflows for bounding-box verification, CSV-to-YOLO conversion, and defect-focused crop generation.
8. Computational benchmarking and a fair same-graph FD-Net V2 / FD-Net V2-KD resource comparison.
9. Industrial review of representative defect samples, categories, labels, and annotations.

## Industrial review

Representative dataset samples, defect categories, class labels, and annotations were reviewed from an industrial garment-manufacturing perspective by professionals from:

**JOYTEX SOURCING LTD., Dhaka, Bangladesh**

This review assessed industrial relevance. **JOYTEX SOURCING LTD. did not create, own, or provide the public source dataset.**

## Limitations

- Only four defect categories are considered.
- The public source dataset does not represent every possible fabric type, weave structure, color, defect severity, or production environment.
- External cross-factory validation has not yet been performed.
- Experiments are primarily based on static images.
- Classification is performed on defect-focused crops rather than a fully integrated end-to-end production pipeline.
- The independent classification test set contains 244 crops.
- The best KD validation macro-F1 (98.04%) was substantially higher than its independent-test macro-F1 (91.73%), indicating that validation performance was optimistic for the held-out test distribution.
- KD hyperparameters were explored on the validation split; stronger future evidence should include repeated random-seed experiments.
- Computational benchmarking was performed on an NVIDIA GeForce RTX 4070 Ti SUPER.
- Power consumption, energy use, and embedded/edge-device performance were not measured.
- Latency and throughput exhibited runtime/order sensitivity in the same-graph KD robustness benchmark.
- Real-world deployment may introduce additional variation in illumination, camera position, fabric texture, and background conditions.

## Future work

Future work should investigate:

- Larger and more diverse fabric-defect datasets
- Additional defect categories
- Cross-dataset and multi-factory validation
- Multiple random-seed experiments with mean ± standard deviation reporting
- Different fabric types and textures
- Variable illumination and camera conditions
- Defect segmentation
- Video-based production-line inspection
- Edge-device deployment
- Quantization and pruning
- FP16 / TensorRT deployment optimization
- Power and energy benchmarking on embedded hardware

Knowledge distillation has already been evaluated in the present repository through FD-Net V2-KD and is therefore **not** listed as an unperformed future-work item.

## Data availability

The public source dataset is available from Roboflow Universe:

https://universe.roboflow.com/yolov7-vdg8u/fabric-defect-detection-jdyz3

The source dataset is subject to its original **CC BY 4.0** license and attribution requirements.

This repository contains processing and experimental code used for the study. Source or derived dataset material should be redistributed only in accordance with the original dataset license and applicable terms.

## Code availability

The code used for data preparation, model training, evaluation, FD-Net V2, FD-Net V2-KD, and computational benchmarking is available in this repository:

https://github.com/niamul-anto/Fabric-Defect-Detection

For publication-quality reproducibility, create a release/tag identifying the exact repository state corresponding to the submitted manuscript.

## Citation

If you use FD-Net V2, FD-Net V2-KD, this repository, the experimental pipeline, or the reported results in academic work, please cite the associated research paper once final publication details are available.

Repository citation metadata is provided in:

```text
CITATION.cff
```

The original Roboflow Universe dataset should also be cited separately.

## Acknowledgments

The authors acknowledge the creators and contributors of the public **Fabric defect detection** dataset hosted on Roboflow Universe.

The authors also acknowledge **JOYTEX SOURCING LTD., Dhaka, Bangladesh** for industrial review of representative fabric-defect samples, categories, labels, and annotations.

JOYTEX SOURCING LTD. was not the source of the public dataset.

## License

The public source dataset is licensed separately under **CC BY 4.0** by its source provider.

The repository code and author-created research materials are distributed according to the license specified in the repository `LICENSE` file. The dataset license and repository-code license are separate.

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
