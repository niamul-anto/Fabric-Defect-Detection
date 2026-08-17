# Dataset Documentation

## FD-Net V2: A Scratch-Trained Lightweight Deep Learning Architecture for Fabric Defect Diagnosis

This document describes the dataset used for the experiments reported in this research project.

The dataset was obtained from the publicly available **Fabric Defect Detection** dataset hosted on Roboflow Universe.

## Dataset Source

**Roboflow Universe:**  
https://universe.roboflow.com/niamul-khan-anto/fabric-defect-detection-jdyz3-bvivw

The original dataset and its annotations are attributed to the respective dataset creators. Users of the dataset should follow the original dataset's applicable license and attribution requirements.

---

## Dataset Overview

The dataset contains images of fabric defects belonging to four defect categories:

1. Cut
2. Hole
3. Stain
4. ThreadError

The dataset was used as the primary experimental dataset for both:

- Object detection
- Defect classification

The source annotations were provided in CSV format containing image filenames, bounding-box coordinates, and defect classes.

The annotation format is:

```text
filename,xmin,ymin,xmax,ymax,class