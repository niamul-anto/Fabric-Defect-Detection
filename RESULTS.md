# Manuscript-Reported Final Results

This file lists only the **final results reported in the manuscript**:

**FD-Net V2: A Scratch-Trained Lightweight Deep Learning Architecture for Fabric Defect Diagnosis**

Development/preliminary experiments such as YOLOv8n or the earlier FD-Net model are intentionally excluded from these final comparison tables.

## Object Detection

Independent-test results:

| Model | Precision (%) | Recall (%) | F1 (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) |
|---|---:|---:|---:|---:|---:|
| YOLOv8m | 75.78 | 50.00 | 60.25 | 64.04 | 39.06 |
| RetinaNet | 70.18 | 65.57 | 67.80 | 71.70 | 48.45 |
| Faster R-CNN | 76.28 | 85.66 | 80.70 | 82.81 | 63.91 |

**Strongest overall localization result:** Faster R-CNN.

## Image Classification

Independent-test results:

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---:|---:|---:|---:|
| ViT-B/16 | 95.08 | 94.28 | 95.55 | 94.84 |
| VGG16-BN | 94.26 | 93.36 | 95.06 | 93.87 |
| EfficientNet-B0 | 92.62 | 91.91 | 93.37 | 92.22 |
| FD-Net V2 | 89.75 | 88.81 | 91.21 | 89.45 |

**Highest classification accuracy:** ViT-B/16 (95.08%).

FD-Net V2 is not presented as the highest-accuracy classifier. Its contribution is a favorable accuracy-resource trade-off while being trained from scratch.

## Computational Characteristics

| Model | Params (M) | Size (MB) | Est. GFLOPs | Peak GPU (MB) |
|---|---:|---:|---:|---:|
| ViT-B/16 | 85.80 | 327.37 | 16.87 | 344.08 |
| VGG16-BN | 134.29 | 512.32 | 15.49 | 2098.24 |
| EfficientNet-B0 | 4.01 | 15.59 | 0.400 | 80.53 |
| FD-Net V2 | 3.07 | 11.88 | 0.899 | 63.38 |

## FP32 Batch-1 Inference

Hardware: NVIDIA GeForce RTX 4070 Ti SUPER

| Model | Latency (ms/image) | Throughput (images/s) |
|---|---:|---:|
| ViT-B/16 | 3.073 | 325.45 |
| VGG16-BN | 2.345 | 426.36 |
| EfficientNet-B0 | 3.683 | 271.55 |
| FD-Net V2 | 3.011 | 332.09 |

## Interpretation

- FD-Net V2 has the lowest parameter count among the evaluated classifiers.
- FD-Net V2 has the smallest serialized model size among the evaluated classifiers.
- FD-Net V2 has the lowest measured peak GPU memory among the evaluated classifiers.
- EfficientNet-B0 has the lowest estimated GFLOPs.
- VGG16-BN has the lowest measured GPU latency under the stated benchmark.
- Therefore, FD-Net V2 should be described as a compact scratch-trained model with a favorable accuracy-resource trade-off, not as the best model on every efficiency metric.
