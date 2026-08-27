# Dataset Documentation

## Source Dataset

The source images and annotations used in this study were obtained from the publicly available **Fabric defect detection** dataset hosted on **Roboflow Universe**.

**Canonical source:**  
https://universe.roboflow.com/yolov7-vdg8u/fabric-defect-detection-jdyz3

Roboflow Universe identifies the project as:

- Title: **Fabric defect detection**
- Task: **Object Detection**
- Author/workspace: **yolov7**
- Images: **2,657**
- Classes: **4**
- Classes: **Cut, Hole, Stain, ThreadError**
- License: **CC BY 4.0**

The authors of the present study **did not independently collect or create the public source dataset**.

## Attribution

The original dataset should be cited according to the citation information supplied on the Roboflow Universe project page.

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

## Experimental Preparation Performed in This Study

The research team performed the following operations for the experiments reported in the manuscript:

1. Image and annotation organization
2. Dataset consistency checking
3. Bounding-box validation
4. Train/validation/test organization
5. CSV annotation processing
6. CSV-to-YOLO conversion for YOLO experiments
7. Defect-focused crop generation for image classification
8. Model-specific resizing, normalization, and training-time augmentation
9. Independent-test evaluation

These steps constitute **experimental preparation and processing**, not original collection or ownership of the source dataset.

## Defect Classes

| Class | Description |
|---|---|
| Cut | Cutting, tearing, or discontinuity in the fabric structure. |
| Hole | An opening or missing region in the fabric. |
| Stain | Localized discoloration or contamination on the fabric surface. |
| ThreadError | Thread-related structural defects such as broken, missing, displaced, or incorrectly woven threads. |

## Object-Detection Split Used in the Study

| Split | Images | Annotation instances |
|---|---:|---:|
| Training | 2,000 | 2,401 |
| Validation | 500 | 669 |
| Test | 157 | 244 |
| **Total** | **2,657** | **3,314** |

## Classification Crops Used in the Study

Defect-focused crops were generated from annotated bounding boxes.

| Split | Cut | Hole | Stain | ThreadError | Total |
|---|---:|---:|---:|---:|---:|
| Training | 548 | 524 | 349 | 974 | 2,395 |
| Validation | 168 | 143 | 161 | 196 | 668 |
| Test | 41 | 62 | 51 | 90 | 244 |
| **Total** | **757** | **729** | **561** | **1,260** | **3,307** |

## Annotation Processing

The study pipeline processed bounding-box annotations in the following CSV form:

```text
filename,xmin,ymin,xmax,ymax,class
```

YOLO-format labels can be generated with:

```bash
python convert_to_yolo.py
```

Classification crops can be generated with:

```bash
python create_classification_dataset.py
```

## Industrial Review

Professionals from **JOYTEX SOURCING LTD., Dhaka, Bangladesh** reviewed representative dataset samples, defect categories, class labels, and annotations from an industrial garment-manufacturing perspective.

This activity served as **domain/industrial review only**.

**JOYTEX SOURCING LTD. did not create, own, collect, or provide the public source dataset used in the experiments.**

## Data Availability and Reuse

Source dataset:  
https://universe.roboflow.com/yolov7-vdg8u/fabric-defect-detection-jdyz3

The source dataset is licensed under **CC BY 4.0**. Any reuse or redistribution of source or derived dataset material must comply with the original license and attribution requirements.

The processing, model-training, evaluation, and benchmarking code for this study is maintained at:

https://github.com/niamul-anto/Fabric-Defect-Detection
