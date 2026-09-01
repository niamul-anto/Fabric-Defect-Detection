# Manuscript-Reported Final Results

This file summarizes only the **final results reported in the manuscript**:

**FD-Net V2: A Scratch-Trained Lightweight Deep Learning Architecture for Fabric Defect Diagnosis**

Developmental and preliminary experiments, including YOLOv8n and earlier FD-Net variants, are intentionally excluded from these final comparison tables.

---

## Object Detection

Independent-test results:

| Model | Precision (%) | Recall (%) | F1 (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) |
|---|---:|---:|---:|---:|---:|
| YOLOv8m | 75.78 | 50.00 | 60.25 | 64.04 | 39.06 |
| RetinaNet | 70.18 | 65.57 | 67.80 | 71.70 | 48.45 |
| Faster R-CNN | 76.28 | 85.66 | 80.70 | 82.81 | 63.91 |

**Strongest overall localization result:** Faster R-CNN, with 82.81% mAP@0.5 and 63.91% mAP@0.5:0.95.

The result should not be interpreted as evidence that two-stage detectors are universally superior; Faster R-CNN was the strongest detector among the evaluated models under the dataset and protocol used in this study.

---

## Image Classification

Independent-test results:

| Model | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|
| ViT-B/16 | 95.08 | 94.28 | 95.55 | 94.84 |
| VGG16-BN | 94.26 | 93.36 | 95.06 | 93.87 |
| EfficientNet-B0 | 92.62 | 91.91 | 93.37 | 92.22 |
| FD-Net V2 | 89.75 | 88.81 | 91.21 | 89.45 |
| FD-Net V2-KD | 92.21 | 91.25 | 92.76 | 91.73 |

**Highest classification accuracy:** ViT-B/16 (95.08%).

FD-Net V2 is not presented as the highest-accuracy classifier. Its contribution is a compact classification architecture trained from random initialization, providing a favorable accuracy-resource trade-off without relying on externally pretrained feature weights.

FD-Net V2-KD uses the same student inference architecture as FD-Net V2 but receives training-time supervision from a fine-tuned ImageNet-pretrained ViT-B/16 teacher.

---

## Knowledge Distillation

Four distillation configurations were compared using validation macro-F1 only:

| Configuration | Temperature T | CE Weight alpha | Best Validation Macro-F1 (%) |
|---|---:|---:|---:|
| T4-A05 | 4 | 0.5 | 97.08 |
| T2-A05 | 2 | 0.5 | 97.18 |
| T4-A07 | 4 | 0.7 | 98.04 |
| T6-A05 | 6 | 0.5 | 97.87 |

**Selected configuration:** `T = 4`, `alpha = 0.7`.

The selected checkpoint was determined using validation performance before evaluation on the independent test set.

### Independent-test improvement

- Accuracy: **89.75% -> 92.21%**
- Accuracy improvement: **+2.46 percentage points**
- Macro-F1: **89.45% -> 91.73%**
- Macro-F1 improvement: **+2.28 percentage points**

The original accuracy gap between ViT-B/16 (95.08%) and scratch-trained FD-Net V2 (89.75%) was 5.33 percentage points.

After knowledge distillation, the remaining gap between ViT-B/16 and FD-Net V2-KD (92.21%) was 2.87 percentage points.

Thus, knowledge distillation closed approximately **46% of the original baseline-to-teacher accuracy gap** without changing the deployed FD-Net V2 architecture.

FD-Net V2-KD remained 0.41 percentage points below EfficientNet-B0 in test accuracy and did not outperform the ViT-B/16 teacher.

---

## Class-Specific Results

The scratch-trained FD-Net V2 correctly classified **219 of 244** independent-test crops.

Its most frequent errors included:

- ThreadError predicted as Cut: 8 cases
- Hole predicted as Cut: 5 cases

FD-Net V2-KD correctly classified **225 of 244** test crops.

Relative to the scratch-trained FD-Net V2 baseline:

- Correct ThreadError predictions increased from 76 to 82.
- Correct Stain predictions increased from 48 to 49.
- Correct Hole predictions remained unchanged at 55.
- Correct Cut predictions decreased from 40 to 39.

Therefore, the aggregate improvement produced by knowledge distillation was class-dependent rather than uniform across all defect categories.

---

## Static Computational Characteristics

| Model | Parameters (M) | Serialized Size (MB) | Estimated GFLOPs |
|---|---:|---:|---:|
| ViT-B/16 | 85.80 | 327.37 | 16.87 |
| VGG16-BN | 134.29 | 512.32 | 15.49 |
| EfficientNet-B0 | 4.01 | 15.59 | 0.400 |
| FD-Net V2 | 3.068 | 11.87 | 0.899 |
| FD-Net V2-KD | 3.068 | 11.87 | 0.899 |

FD-Net V2 and FD-Net V2-KD each contain exactly **3,068,448 trainable parameters**.

Knowledge distillation changes the learned student weights but does not change the deployed student architecture. Therefore, FD-Net V2 and FD-Net V2-KD have identical:

- parameter count,
- serialized model size,
- estimated GFLOPs,
- inference architecture.

EfficientNet-B0 has the lowest estimated GFLOPs among the evaluated classifiers.

FD-Net V2 has the lowest parameter count and smallest serialized weight footprint among the evaluated classifiers.

---

## Fair Same-Implementation FP32 Benchmark

Hardware: **NVIDIA GeForce RTX 4070 Ti SUPER**

Input: **224 x 224**

Batch size: **1**

Both FD-Net V2 and FD-Net V2-KD were evaluated using the same canonical inference implementation.

| Benchmark Order | Model | Latency (ms/image) | Throughput (images/s) | Peak GPU Memory (MB) | GFLOPs |
|---|---|---:|---:|---:|---:|
| Baseline-first | FD-Net V2 | 2.509 | 398.64 | 27.43 | 0.899 |
| Baseline-first | FD-Net V2-KD | 2.490 | 401.63 | 27.43 | 0.899 |
| KD-first | FD-Net V2 | 2.509 | 398.61 | 27.43 | 0.899 |
| KD-first | FD-Net V2-KD | 4.174 | 239.56 | 27.43 | 0.899 |

The two models had identical parameter count, serialized size, estimated GFLOPs, and measured peak allocated GPU memory under the same canonical implementation.

Latency and throughput for the KD checkpoint were sensitive to benchmark order. Therefore, the manuscript does **not** claim that knowledge distillation caused either a speedup or slowdown.

Deployment claims are based on stable architectural properties and on the fact that the ViT-B/16 teacher is not required during student inference.

---

## Interpretation

- ViT-B/16 achieved the highest independent-test classification accuracy.
- Faster R-CNN achieved the strongest object-detection performance under the evaluated protocol.
- FD-Net V2 has the lowest parameter count among the evaluated classifiers.
- FD-Net V2 has the smallest serialized model size among the evaluated classifiers.
- EfficientNet-B0 has the lowest estimated GFLOPs.
- FD-Net V2-KD improves the scratch-trained FD-Net V2 baseline by 2.46 percentage points in accuracy and 2.28 percentage points in macro-F1.
- FD-Net V2 and FD-Net V2-KD use the same deployed inference architecture.
- Knowledge distillation does not add inference-time teacher parameters or operations.
- No latency improvement is attributed to knowledge distillation because GPU timing was benchmark-order sensitive.
- FD-Net V2 should therefore be described as a compact scratch-trained classifier with a favorable accuracy-resource trade-off, rather than as the best model on every predictive or computational metric.
- FD-Net V2-KD should be described separately as a teacher-guided training extension; although the student was randomly initialized, its training used knowledge from an ImageNet-pretrained ViT-B/16 teacher.

---

## Reproducibility Note

The final manuscript results were obtained using fixed training, validation, and independent-test partitions. Model and hyperparameter selection were performed using the validation subset, and the held-out test subset was reserved for final evaluation.

The best FD-Net V2-KD validation macro-F1 (98.04%) was higher than its independent-test macro-F1 (91.73%). Consequently, the manuscript explicitly recommends future evaluation using multiple random seeds, repeated splits, and external datasets rather than making broader generalization claims from a single split.