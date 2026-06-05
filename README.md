# Retinal OCT Classification
# Retinal OCT Binary Classification: Transfer Learning vs. Fine-Tuning


**Binary classification of retinal OCT images (Healthy vs. Disease) using EfficientNet-B0 with controlled data leakage removal and GMLP-inspired documentation.**

This project benchmarks three training strategies—Baseline Transfer Learning, CLAHE Preprocessing, and Full Fine-Tuning—on the Kermany2018 dataset, with strict control for data leakage between train and test sets.

---

## 1. Overview

Optical Coherence Tomography (OCT) is a non-invasive imaging technique critical for diagnosing retinal diseases such as Diabetic Macular Edema (DME), Choroidal Neovascularization (CNV), and Drusen. This project implements a binary classification pipeline (Healthy vs. Pathology) to demonstrate reproducible medical AI workflows, including:
- **Data Leakage Control:** MD5-based deduplication between train/test splits.
- **Class Imbalance Handling:** WeightedRandomSampler to address the dominance of "Disease" class.
- **Conservative Augmentations:** Geometric transforms (Flip, Rotate 5°) chosen to preserve anatomical structure, following MIDRC principles.
- **Interpretability:** Grad-CAM visualization to verify that the model attends to clinically relevant regions rather than artifacts.

**Key Finding:** Full fine-tuning of the backbone significantly outperforms frozen feature extraction and preprocessing adjustments, achieving near-perfect separation on the benchmark test set.

---

## 2. Dataset

**Source:** [Kermany et al., 2018](https://data.mendeley.com/datasets/rscbjbr9sj/2) (OCT2017)  
**Classes (Original):** CNV, DME, DRUSEN, NORMAL  
**Task (Binary):** 
- **Disease (1):** CNV, DME, DRUSEN merged.
- **Healthy (0):** NORMAL.

**Preprocessing & Quality Control:**
- **Leakage Removal:** Identified exact duplicates between `train` and `test` via MD5 hashing and filtered them prior to training.
- **Resolution:** Resized to 224×224 (EfficientNet-B0 input).
- **Color Space:** Grayscale (L) converted to 3-channel RGB for ImageNet compatibility.
- **Augmentation:** HorizontalFlip(p=0.5), RandomRotation(5°). Aggressive distortions excluded to prevent anatomical corruption.

---

## 3. Methodology

### Model Architecture
- **Backbone:** `efficientnet_b0` (via `timm`).
- **Head:** Single linear layer (binary classification with BCEWithLogitsLoss).

### Experimental Setup
Three independent training runs were conducted:

1.  **Baseline:** Frozen backbone (ImageNet weights), training only the classifier head (LR=3e-4).
2.  **CLAHE:** Applied Contrast Limited Adaptive Histogram Equalization (clipLimit=2.0) before the baseline pipeline to test if local contrast enhancement improves feature extraction.
3.  **Fine-Tuning:** Loaded Baseline weights, unfroze all layers, and continued training with a low learning rate (LR=1e-5) to adapt the feature space to OCT-specific textures.

### Metrics
Chosen for medical screening relevance (imbalanced data):
- **F1-Score:** Balance between Precision and Recall.
- **ROC-AUC:** Discriminative ability across thresholds.
- **PR-AUC (Precision-Recall):** Critical for imbalanced datasets where high precision is required to avoid false positive diagnoses.

---

## 4. Results

| Model | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: |
| **Baseline** | 0.919 | 0.966 | 0.989 |
| **CLAHE** | 0.928 | 0.965 | 0.989 |
| **Fine-Tuning** | **0.997** | **0.9997** | **0.9999** |

**Interpretation:**
- **Baseline** already achieves strong performance, confirming the power of transfer learning for OCT.
- **CLAHE** provides a marginal F1 improvement but negligible gain in AUC metrics, suggesting it aids threshold-specific classification but not overall separability.
- **Fine-Tuning** dominates, indicating that adapting the convolutional filters to OCT-specific patterns is far more effective than preprocessing or shallow transfer.

**Caveat:** These metrics reflect performance on the Kermany2018 benchmark. The dataset is relatively clean and homogeneous compared to real-world clinical data (varying devices, patient populations, and acquisition artifacts). **External validation is required before clinical deployment.**

### Interpretability (Grad-CAM)
On pathological images (positive class), the model correctly localizes attention to regions of fluid accumulation and structural disruption. On healthy images (negative class), activations are diffuse, indicating the model scans for non-existent anomalies before concluding "Healthy." This behavior is expected and clinically acceptable.

---

## 5. Documentation Standards (GMLP / MIDRC)

This project adheres to principles outlined in the FDA's Good Machine Learning Practice (GMLP) and the Medical Imaging and Data Resource Center (MIDRC) guidelines:

- **Dataset Integrity:** Explicit handling of data leakage via hashing ensures reported metrics reflect true generalization, not memorization.
- **Train/Val/Test Splitting:** Stratified sampling (15% validation) with class balance preservation; test set strictly isolated post-deduplication.
- **Augmentation Justification:** Transformations limited to clinically safe operations (rotation, flip) to avoid generating anatomically implausible training examples.
- **Metric Selection:** PR-AUC prioritized alongside ROC-AUC due to class imbalance, aligning with screening task requirements.

---

## 6. Repository Structure

```text
retinal-oct-classification/
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore                # Excludes data/cache/model weights
├── retinal_oct_classification.ipynb  # Main notebook (EDA + Training + Evaluation)
│
├── assets/                   # Figures for README (ROC curves, Grad-CAM, etc.)
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── gradcam_samples.png
│
└── results/                  # Serialized metrics (optional)
    ├── metrics_summary.csv
    └── training_histories.json
