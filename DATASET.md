# Self-Developed Fabric Defect Dataset

## Overview

This repository uses a **self-developed fabric defect dataset** created specifically for the research project titled:

**“Fabric Defect Diagnosis Using Deep Learning with a Scratch-Trained Lightweight FD-Net V2.”**

The dataset was **independently collected, prepared, cleaned, categorized, and manually annotated by the authors**.

No publicly available fabric-defect dataset was used as the primary experimental dataset for the reported results.

The dataset supports two computer-vision tasks:

* **Object Detection**
* **Image Classification**

---

## Dataset Ownership

The dataset was developed by the following researchers from the Department of Computer Science and Engineering, Brac University:

* Md Niamul Islam Khan
* Md Al Mamunur Rashid Emon
* Miskatunnisa Labonno
* Rubiya Tasfi Bidisha

The authors were responsible for:

* Image collection
* Image screening and cleaning
* Defect categorization
* Class-label preparation
* Bounding-box annotation
* Dataset splitting
* Classification crop generation
* Experimental dataset preparation

The dataset was not supplied by JOYTEX SOURCING LTD. or by any public dataset repository.

---

## Industrial Review

Representative images, defect categories, class labels, and annotations were reviewed from an industrial perspective by professionals from:

**JOYTEX SOURCING LTD.**
Dhaka, Bangladesh

This review was conducted to assess whether the selected samples, defect categories, and annotations were consistent with commonly observed fabric defects in practical garment and textile inspection.

The role of JOYTEX SOURCING LTD. was limited to **industrial review and verification**.

**The company did not provide or create the dataset.**

---

## Defect Classes

The dataset contains four fabric defect categories:

1. **Cut**
2. **Hole**
3. **Stain**
4. **ThreadError**

### Cut

A visible tear, cut, or discontinuity in the fabric structure.

### Hole

An opening in the fabric caused by missing or damaged fabric material.

### Stain

Localized discoloration or contamination on the fabric surface without necessarily creating a physical opening or tear.

Typical examples may include:

* Oil marks
* Ink marks
* Chemical marks
* Dye irregularities
* Dirty spots

### ThreadError

A structural thread-related defect involving broken, missing, displaced, or incorrectly woven threads.

---

## Dataset Statistics

### Object Detection Dataset

The object-detection dataset contains a total of:

**3,314 annotated defect instances**

| Split      | Defect Instances | Purpose                                         |
| ---------- | ---------------: | ----------------------------------------------- |
| Training   |            2,401 | Model training and optimization                 |
| Validation |              669 | Performance monitoring and checkpoint selection |
| Test       |              244 | Independent final evaluation                    |
| **Total**  |        **3,314** | Complete annotated detection dataset            |

The original fabric images and their bounding-box annotations were used for object-detection experiments.

---

## Classification Dataset

Defect-focused classification images were generated from the annotated defect regions.

The resulting classification dataset contains:

**3,307 image crops**

| Split      |     Cut |    Hole |   Stain | ThreadError |     Total |
| ---------- | ------: | ------: | ------: | ----------: | --------: |
| Training   |     548 |     524 |     349 |         974 |     2,395 |
| Validation |     168 |     143 |     161 |         196 |       668 |
| Test       |      41 |      62 |      51 |          90 |       244 |
| **Total**  | **757** | **729** | **561** |   **1,260** | **3,307** |

The classification task uses defect-focused crops corresponding to the annotated bounding-box regions.

---

## Annotation Protocol

Each visible defect used for object detection was assigned:

* A defect class label
* A rectangular bounding box

The annotation process followed a consistent class-labeling and bounding-box protocol.

The annotations served two purposes:

1. Ground-truth labels for object-detection training and evaluation
2. Spatial references for generating defect-focused classification crops

---

## Dataset Preparation Workflow

The dataset preparation process consisted of the following stages:

```text
Image Collection
      ↓
Image Review and Cleaning
      ↓
Defect Categorization
      ↓
Bounding-Box Annotation
      ↓
Train / Validation / Test Split
      ↓
 ┌─────────────────────────────┐
 │                             │
 ↓                             ↓
Object Detection         Classification Crop
Dataset Preparation      Generation
 │                             │
 ↓                             ↓
Detection Models         Classification Models
```

---

## Data Cleaning

Before training, images were reviewed to reduce dataset-quality problems.

The cleaning process considered:

* Corrupted images
* Duplicate images
* Unclear samples
* Incorrectly labeled samples
* Inconsistent defect labels
* Poor-quality samples unsuitable for training

---

## Data Splitting

The dataset was divided into:

* Training set
* Validation set
* Independent test set

The training data were used for model optimization.

The validation data were used for:

* Performance monitoring
* Learning-rate adjustment where applicable
* Early stopping
* Best-checkpoint selection

The independent test data were reserved for final model evaluation.

The test subset was not used for model training or best-checkpoint selection.

---

## Classification Image Preparation

Classification images were generated by cropping annotated defect regions from the source images.

The classification images were resized to:

**224 × 224 pixels**

Training images were subjected to model-specific augmentation.

Validation and test images were not randomly augmented during evaluation.

---

## Object Detection Input Preparation

Different detection models used architecture-specific image preprocessing.

### YOLOv8m

Input resolution:

**960 × 960 pixels**

### RetinaNet

The image aspect ratio was preserved.

* Shorter side: approximately 960 pixels
* Longer side: capped at 1280 pixels

### Faster R-CNN

The image aspect ratio was preserved.

* Shorter side: approximately 960 pixels
* Longer side: capped at 1280 pixels

---

## Class-Imbalance Handling

The dataset contains an unequal number of samples across the four defect classes.

For classification training, class-imbalance handling methods were applied according to the individual model-training pipeline.

The independent validation and test distributions were not artificially balanced for final evaluation.

---

## Models Evaluated Using the Dataset

### Object Detection

* YOLOv8m
* RetinaNet
* Faster R-CNN

Additional YOLOv8n experiments were conducted during model development and are retained in the repository.

### Image Classification

* ViT-B/16
* VGG16-BN
* EfficientNet-B0
* FD-Net V1
* Proposed FD-Net V2

---

## Proposed FD-Net V2

FD-Net V2 was developed specifically as a lightweight classifier for this dataset.

Unlike the evaluated pretrained classifiers, FD-Net V2 was:

* Randomly initialized
* Trained entirely from scratch
* Trained without ImageNet-pretrained features
* Optimized directly using the developed fabric-defect dataset

The proposed FD-Net V2 achieved:

* Accuracy: **89.75%**
* Precision: **88.81%**
* Recall: **91.21%**
* F1-score: **89.45%**
* Trainable parameters: **3,068,448**

---

## Intended Research Use

The dataset was developed for research on:

* Automated fabric defect detection
* Fabric defect classification
* Lightweight deep neural networks
* Object detection
* Computer vision for textile inspection
* Model accuracy-complexity analysis
* Resource-efficient deep learning

---

## Limitations

The current dataset has several limitations:

* Only four defect categories are included.
* The dataset does not yet represent all possible fabric types, colors, weave patterns, defect severities, or production conditions.
* External cross-factory validation has not yet been completed.
* The dataset primarily contains static images.
* Continuous production-line video data are not included.
* Multiple overlapping defects and unseen defect categories require further study.

Future versions may include more factories, defect classes, fabrics, cameras, illumination conditions, and industrial production scenarios.

---

## Dataset Availability

The complete image dataset is not currently distributed directly through this GitHub repository.

The repository mainly contains:

* Training scripts
* Evaluation scripts
* Benchmarking scripts
* Dataset-preparation utilities
* Experimental results
* Model configuration information

Researchers interested in the dataset should contact the authors regarding availability and permitted research use.

---

## Dataset Citation

If this dataset is used in academic research, please cite the associated research paper once the publication information becomes available.

A formal repository citation file will also be provided through:

```text
CITATION.cff
```

after the final publication metadata become available.

---

## Contact

### Md Niamul Islam Khan

Department of Computer Science and Engineering
Brac University
Email: [niamul.islam.khan@g.bracu.ac.bd](mailto:niamul.islam.khan@g.bracu.ac.bd)

### Md Al Mamunur Rashid Emon

Department of Computer Science and Engineering
Brac University
Email: [md.al.mamunur.rashid.emon@g.bracu.ac.bd](mailto:md.al.mamunur.rashid.emon@g.bracu.ac.bd)

### Miskatunnisa Labonno

Department of Computer Science and Engineering
Brac University
Email: [miskatunnisa.labonno@g.bracu.ac.bd](mailto:miskatunnisa.labonno@g.bracu.ac.bd)

### Rubiya Tasfi Bidisha

Department of Computer Science and Engineering
Brac University
Email: [rubiya.tasfi.bidisha@g.bracu.ac.bd](mailto:rubiya.tasfi.bidisha@g.bracu.ac.bd)
