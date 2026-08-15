# Fabric Defect Diagnosis Using Deep Learning with a Scratch-Trained Lightweight FD-Net V2

A deep learning-based framework for **fabric defect detection and classification** using a **self-developed and manually annotated fabric defect dataset**. The project evaluates multiple object-detection and image-classification architectures and proposes **FD-Net V2**, a compact classifier trained entirely from scratch without external pretrained features.

---

## Project Overview

Automated fabric inspection is important for maintaining product quality in textile manufacturing. Fabric defects may be small, irregular, low-contrast, or visually similar to normal fabric texture, making manual inspection difficult and inconsistent.

This research investigates two complementary computer-vision tasks:

* **Object Detection:** identifying both the defect category and its location using bounding boxes.
* **Image Classification:** identifying the defect category from defect-focused image crops.

The study evaluates established deep learning architectures under a common experimental framework and introduces the proposed **FD-Net V2 lightweight classifier**.

The four investigated fabric defect classes are:

1. **Cut**
2. **Hole**
3. **Stain**
4. **ThreadError**

---

## Self-Developed Dataset

The dataset used in this research was **independently collected, prepared, categorized, cleaned, and manually annotated by the authors**.

No publicly available fabric-defect dataset was used as the primary experimental dataset.

### Dataset Summary

* **Number of defect classes:** 4
* **Object-detection defect instances:** 3,314
* **Classification defect crops:** 3,307
* **Classes:** Cut, Hole, Stain, ThreadError
* **Annotation type:** Bounding boxes
* **Tasks supported:** Object detection and image classification

### Object-Detection Dataset

| Split      | Defect Instances | Purpose                                         |
| ---------- | ---------------: | ----------------------------------------------- |
| Training   |            2,401 | Model training and optimization                 |
| Validation |              669 | Performance monitoring and checkpoint selection |
| Test       |              244 | Independent final evaluation                    |
| **Total**  |        **3,314** | Complete annotated detection dataset            |

### Classification Dataset

| Split      |     Cut |    Hole |   Stain | ThreadError |     Total |
| ---------- | ------: | ------: | ------: | ----------: | --------: |
| Training   |     548 |     524 |     349 |         974 |     2,395 |
| Validation |     168 |     143 |     161 |         196 |       668 |
| Test       |      41 |      62 |      51 |          90 |       244 |
| **Total**  | **757** | **729** | **561** |   **1,260** | **3,307** |

Defect-focused classification crops were generated from the manually annotated bounding-box regions.

Representative samples, defect categories, and annotations were also reviewed from an industrial perspective by professionals from **JOYTEX SOURCING LTD., Dhaka, Bangladesh**.

**JOYTEX SOURCING LTD. did not provide the dataset. The dataset was independently developed by the authors.**

---

## Dataset Preparation Pipeline

The dataset preparation process included:

1. Fabric image collection
2. Image cleaning and quality review
3. Class-label verification
4. Bounding-box annotation
5. Train/validation/test splitting
6. Detection-specific preprocessing
7. Classification crop generation
8. Training-only data augmentation
9. Class-imbalance handling
10. Independent test evaluation

For classification, images were resized to:

```text
224 × 224
```

For object detection:

* **YOLOv8m:** 960 × 960 input
* **RetinaNet:** shorter side resized to 960 px, longer side capped at 1280 px
* **Faster R-CNN:** shorter side resized to 960 px, longer side capped at 1280 px

---

## Models Evaluated

### Object Detection

The final study evaluates:

* **YOLOv8m**
* **RetinaNet**
* **Faster R-CNN**

These architectures represent different detection strategies:

* YOLOv8m — one-stage anchor-free detection
* RetinaNet — focal-loss-based dense detection
* Faster R-CNN — two-stage proposal-based detection

Additional YOLOv8n experiments are retained in the repository as development experiments but are not treated as a principal model in the final paper comparison.

---

## Image Classification

The evaluated classification architectures are:

* **ViT-B/16**
* **VGG16-BN**
* **EfficientNet-B0**
* **FD-Net V1**
* **Proposed FD-Net V2**

### Initialization Strategy

| Model           | Initialization                                   |
| --------------- | ------------------------------------------------ |
| ViT-B/16        | ImageNet pretrained                              |
| VGG16-BN        | ImageNet pretrained                              |
| EfficientNet-B0 | ImageNet pretrained                              |
| FD-Net V1       | Random initialization                            |
| **FD-Net V2**   | **Random initialization / trained from scratch** |

---

# Proposed FD-Net V2

The primary proposed contribution of this research is **FD-Net V2**, a lightweight fabric-defect classification architecture designed and trained from scratch using only the developed fabric-defect dataset.

FD-Net V2 does **not** use ImageNet or any other external pretrained feature representations.

## Main Architectural Components

FD-Net V2 combines:

* Standard convolutional feature extraction
* Inverted Residual Blocks
* Depthwise convolution
* Residual shortcut connections
* Channel attention
* Spatial attention
* Combined channel-spatial attention
* Global average pooling
* Fully connected classification head

The conceptual feature flow is:

```text
Input Image
    ↓
Convolutional Stem
    ↓
Standard Convolution Stage
    ↓
Inverted Residual Blocks
    ↓
Depthwise Feature Extraction
    ↓
Selective Channel-Spatial Attention
    ↓
Deeper Feature Refinement
    ↓
Global Average Pooling
    ↓
Fully Connected Classifier
    ↓
Cut / Hole / Stain / ThreadError
```

### FD-Net V2 Configuration

* Input size: **224 × 224 × 3**
* Number of classes: **4**
* Total parameters: **3,068,448**
* Parameter count: **≈ 3.07 million**
* Training initialization: **Random**
* Optimizer: **AdamW**
* Initial learning rate: **3 × 10⁻⁴**
* Weight decay: **1 × 10⁻⁴**
* Batch size: **16**
* Maximum epochs: **100**
* Early-stopping patience: **20**
* Best checkpoint criterion: **Validation F1-score**
* Random seed: **42**

---

# Model Performance

## Classification Results

The following results were obtained on the independent classification test set.

| Model                  |   Accuracy |  Precision |     Recall |   F1-score |
| ---------------------- | ---------: | ---------: | ---------: | ---------: |
| ViT-B/16               | **95.08%** | **94.28%** | **95.55%** | **94.84%** |
| VGG16-BN               |     94.26% |     93.36% |     95.06% |     93.87% |
| EfficientNet-B0        |     92.62% |     91.91% |     93.37% |     92.22% |
| **Proposed FD-Net V2** | **89.75%** | **88.81%** | **91.21%** | **89.45%** |
| FD-Net V1              |     62.70% |     62.25% |     61.55% |     60.54% |

### Best Classification Model

**ViT-B/16** achieved the highest classification performance:

* Accuracy: **95.08%**
* Precision: **94.28%**
* Recall: **95.55%**
* F1-score: **94.84%**

FD-Net V2 does not aim to replace ViT-B/16 in terms of absolute classification accuracy. Its contribution lies in providing a substantially more compact **scratch-trained lightweight alternative**.

---

## Object Detection Results

| Model            |  Precision |     Recall |   F1-score |    mAP@0.5 | mAP@0.5:0.95 |
| ---------------- | ---------: | ---------: | ---------: | ---------: | -----------: |
| YOLOv8m          |     75.78% |     50.00% |     60.25% |     64.04% |       39.06% |
| RetinaNet        |     70.18% |     65.57% |     67.80% |     71.70% |       48.45% |
| **Faster R-CNN** | **76.28%** | **85.66%** | **80.70%** | **82.81%** |   **63.91%** |

### Best Detection Model

**Faster R-CNN** achieved the strongest overall detection performance:

* Precision: **76.28%**
* Recall: **85.66%**
* F1-score: **80.70%**
* mAP@0.5: **82.81%**
* mAP@0.5:0.95: **63.91%**

The two-stage proposal and bounding-box refinement mechanism provided the strongest localization performance for the investigated fabric defects.

---

## YOLOv8m Per-Class Detection Performance

| Defect Class | mAP@0.5 |
| ------------ | ------: |
| Cut          |  85.87% |
| Hole         |  69.08% |
| Stain        |  60.34% |
| ThreadError  |  40.92% |

YOLOv8m achieved its highest class-wise mAP@0.5 for **Cut**, while **ThreadError** was the most challenging defect category under the evaluated configuration.

---

# FD-Net V2 Independent Test Performance

The proposed FD-Net V2 achieved:

* Accuracy: **89.75%**
* Precision: **88.81%**
* Recall: **91.21%**
* F1-score: **89.45%**
* Best validation F1-score: **97.65%**
* Best checkpoint: **Epoch 98**

### Class-Wise FD-Net V2 Performance

| Class       | Precision | Recall | F1-score | Support |
| ----------- | --------: | -----: | -------: | ------: |
| Cut         |    74.07% | 97.56% |   84.21% |      41 |
| Hole        |    94.83% | 88.71% |   91.67% |      62 |
| Stain       |    88.89% | 94.12% |   91.43% |      51 |
| ThreadError |    97.44% | 84.44% |   90.48% |      90 |

---

# Lightweight Computational Benchmark

To evaluate the lightweight characteristics of FD-Net V2, classification models were benchmarked under the same inference conditions.

### Benchmark Conditions

* GPU: **NVIDIA GeForce RTX 4070 Ti SUPER**
* Input size: **224 × 224**
* Batch size: **1**
* Precision: **FP32**
* Warm-up runs: **100**
* Timed inference runs: **1,000**
* Framework: **PyTorch**
* CUDA: **12.6**

| Model                  |   Accuracy | Parameters |   Model Size | Estimated GFLOPs |      Latency |       Throughput | Peak GPU Memory |
| ---------------------- | ---------: | ---------: | -----------: | ---------------: | -----------: | ---------------: | --------------: |
| ViT-B/16               | **95.08%** |     85.80M |    327.37 MB |            16.87 |     3.073 ms |     325.45 img/s |       344.08 MB |
| VGG16-BN               |     94.26% |    134.29M |    512.32 MB |            15.49 | **2.345 ms** | **426.36 img/s** |      2098.24 MB |
| EfficientNet-B0        |     92.62% |      4.01M |     15.59 MB |        **0.400** |     3.683 ms |     271.55 img/s |        80.53 MB |
| **Proposed FD-Net V2** | **89.75%** |  **3.07M** | **11.88 MB** |            0.899 |     3.011 ms |     332.09 img/s |    **63.38 MB** |

> **Note:** GFLOPs were estimated using the same `fvcore`-based analysis procedure. Some framework-level operations are not counted by the analyzer; therefore, GFLOPs values should be interpreted as comparative estimates rather than exact operation counts.

---

# Lightweight Contribution of FD-Net V2

The proposed FD-Net V2 provides a **compact, dataset-specific, and fully scratch-trained solution** for fabric defect classification.

Despite using no external pretrained features, FD-Net V2 achieved:

* **89.75% classification accuracy**
* **3.07 million trainable parameters**
* **11.88 MB model footprint**
* **63.38 MB peak GPU memory**
* **3.011 ms/image FP32 batch-1 latency**
* **332.09 images/s throughput**

Among the evaluated classification architectures, FD-Net V2 achieved the:

* **Lowest parameter count**
* **Smallest model-storage requirement**
* **Lowest measured peak GPU-memory usage**

Compared with ViT-B/16, FD-Net V2 uses approximately:

* **28× fewer parameters**
* **27.6× smaller model storage**

Compared with VGG16-BN, FD-Net V2 uses approximately:

* **44× fewer parameters**
* **More than 43× smaller model storage**

Even compared with efficiency-oriented EfficientNet-B0, FD-Net V2 remains smaller in parameter count, model size, and measured GPU-memory usage.

The contribution of FD-Net V2 therefore lies **not in achieving the highest absolute classification accuracy**, but in providing a compact, dataset-specific, scratch-trained architecture with a favorable balance between classification performance and resource requirements.

---

# Key Findings

The experiments indicate that no single model is optimal for every fabric inspection requirement.

### Best Defect Localization

**Faster R-CNN**

* Best mAP@0.5
* Best mAP@0.5:0.95
* Best recall
* Best overall detection F1-score

### Best Image Classification

**ViT-B/16**

* Highest classification accuracy
* Highest classification F1-score

### Proposed Lightweight Solution

**FD-Net V2**

* Trained entirely from scratch
* Lowest parameter count among evaluated classifiers
* Smallest model footprint
* Lowest measured peak GPU-memory usage
* Competitive classification performance

---

# Repository Structure

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
├── check_classes.py
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
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/niamul-anto/Fabric-Defect-Detection.git
cd Fabric-Defect-Detection
```

## 2. Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

# Dataset Preparation

The full self-developed image dataset is not included in the current repository.

After placing the source dataset in the appropriate local directory, classification crops can be generated using:

```bash
python create_classification_dataset.py
```

YOLO-format annotations can be prepared using:

```bash
python convert_to_yolo.py
```

Paths in the training scripts may need to be updated according to the local dataset location.

---

# Training

## Proposed FD-Net V2

```bash
python train_fdnet_v2.py
```

## EfficientNet-B0

```bash
python train_efficientnet_b0_final.py
```

## VGG16-BN

```bash
python train_vgg16_classification.py
```

## ViT-B/16

```bash
python train_vit_b16_final.py
```

## Faster R-CNN

```bash
python train_fasterrcnn.py
```

## RetinaNet

```bash
python train_retinanet.py
```

## YOLOv8m

```bash
python train_yolov8m.py
```

---

# Evaluation

## Faster R-CNN

```bash
python evaluate_fasterrcnn.py
```

## RetinaNet

```bash
python evaluate_retinanet.py
```

## YOLOv8m

```bash
python evaluate_yolov8m.py
```

Optional YOLOv8m threshold analysis:

```bash
python evaluate_yolov8m_thresholds.py
```

---

# Computational Benchmarking

The following scripts reproduce the classification-model complexity and inference benchmarking procedure:

## FD-Net V2

```bash
python benchmark_fdnet_v2.py
```

## ViT-B/16

```bash
python benchmark_vit_b16.py
```

## VGG16-BN

```bash
python benchmark_vgg16_bn.py
```

## EfficientNet-B0

```bash
python benchmark_efficientnet_b0.py
```

The benchmarking scripts report:

* Parameter count
* Serialized model size
* Estimated GFLOPs
* FP32 inference latency
* Throughput
* Peak GPU memory usage

---

# Experimental Environment

The main experiments were conducted using:

| Component        | Specification                    |
| ---------------- | -------------------------------- |
| CPU              | Intel Core i7-14700              |
| GPU              | NVIDIA GeForce RTX 4070 Ti SUPER |
| GPU Memory       | 16 GB GDDR6X                     |
| RAM              | 64 GB DDR5                       |
| Storage          | 1 TB NVMe SSD                    |
| Operating System | Windows 11                       |
| Python           | 3.12.3                           |
| CUDA             | 12.6                             |
| Ultralytics      | 8.4.110                          |

PyTorch and Torchvision were used as the primary deep-learning libraries.

---

# Reproducibility

The experiments were designed using:

```text
Random Seed = 42
```

Training, validation, and independent test subsets were separated before final evaluation.

For classification models, the best checkpoint was selected using the **validation F1-score** and subsequently evaluated on the independent test set.

The independent test set was not used for:

* Model training
* Early stopping
* Hyperparameter optimization
* Best-checkpoint selection

---

# Research Contributions

The principal contributions of this project are:

1. Development of a **self-collected and manually annotated four-class fabric defect dataset** suitable for both object detection and image classification.

2. Controlled evaluation of three principal object-detection architectures:

   * YOLOv8m
   * RetinaNet
   * Faster R-CNN

3. Controlled comparison of three established pretrained classifiers:

   * ViT-B/16
   * VGG16-BN
   * EfficientNet-B0

4. Development of the **scratch-trained lightweight FD-Net V2 classifier**.

5. Integration of:

   * Inverted residual learning
   * Depthwise convolution
   * Residual shortcuts
   * Channel attention
   * Spatial attention

6. Evaluation of both predictive performance and model complexity.

7. Identification of task-specific model choices:

   * Faster R-CNN for strong localization
   * ViT-B/16 for maximum classification accuracy
   * FD-Net V2 as a compact scratch-trained lightweight alternative

---

# Limitations

The current research has several limitations:

* Only four defect classes were considered.
* External cross-dataset validation has not yet been performed.
* Models were primarily evaluated using static images.
* The classification task uses defect-focused image crops.
* The models were not evaluated on embedded or edge-AI hardware.
* Energy and power consumption were not directly measured.
* Most final model results were obtained from a single fixed-seed training run.

Although inference latency, throughput, and GPU memory were benchmarked on the NVIDIA RTX 4070 Ti SUPER, actual real-time edge-device suitability requires further deployment testing.

---

# Future Work

Future extensions may include:

* Larger multi-factory datasets
* Additional fabric defect classes
* External dataset validation
* Multi-seed repeated experiments
* Statistical confidence intervals
* Semantic and instance segmentation
* Knowledge distillation
* Model pruning
* Quantization
* Embedded/edge-device deployment
* Power and energy benchmarking
* Video-based production-line inspection
* Cross-fabric and cross-camera evaluation

---

# Authors

* **Md Niamul Islam Khan**
  Department of Computer Science and Engineering
  Brac University
  Email: [niamul.islam.khan@g.bracu.ac.bd](mailto:niamul.islam.khan@g.bracu.ac.bd)

* **Md Al Mamunur Rashid Emon**
  Department of Computer Science and Engineering
  Brac University
  Email: [md.al.mamunur.rashid.emon@g.bracu.ac.bd](mailto:md.al.mamunur.rashid.emon@g.bracu.ac.bd)

* **Miskatunnisa Labonno**
  Department of Computer Science and Engineering
  Brac University
  Email: [miskatunnisa.labonno@g.bracu.ac.bd](mailto:miskatunnisa.labonno@g.bracu.ac.bd)

* **Rubiya Tasfi Bidisha**
  Department of Computer Science and Engineering
  Brac University
  Email: [rubiya.tasfi.bidisha@g.bracu.ac.bd](mailto:rubiya.tasfi.bidisha@g.bracu.ac.bd)

---

# Academic Supervision

**Supervisor**
Md. Tanzim Reza
Senior Lecturer
Department of Computer Science and Engineering
Brac University

**Co-Supervisor**
Dr. Md. Golam Rabiul Alam
Professor
Department of Computer Science and Engineering
Brac University

---

# Acknowledgment

The authors acknowledge **JOYTEX SOURCING LTD., Dhaka, Bangladesh**, for reviewing representative fabric-defect samples, defect categories, and annotations from an industrial perspective.

The dataset itself was independently collected, prepared, and annotated by the authors.

---

# Citation

If you use this repository or the proposed FD-Net V2 architecture in academic work, please cite the corresponding research paper once its publication information becomes available.

A formal `CITATION.cff` file will be added to the repository for citation support.

---

# License

A software and dataset licensing statement will be added before the public research release.

Please contact the authors before redistributing the dataset or using it for commercial purposes.
