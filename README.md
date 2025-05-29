# Software Fault Prediction: Classical and Hybrid Classification Approaches

## Overview

This repository provides an academic implementation and evaluation of classical supervised classification algorithms and a hybrid approach for software fault prediction, inspired by the paper:

> **A Hybrid Approach Based on k-Nearest Neighbors and Decision Tree for Software Fault Prediction**  
> [Authors: S. K. Dubey, A. Rana, et al.]

The project aims to reproduce and compare the performance of classical classifiers (Decision Tree, k-Nearest Neighbors, Support Vector Machine, Random Forest) on PROMISE software defect datasets, and to analyze results against those reported in the referenced research.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments & Results](#experiments--results)
- [Comparison with Literature](#comparison-with-literature)
- [References](#references)
- [License](#license)

---

---

## Dataset

This project utilizes the **PROMISE NASA software defect datasets** (e.g., PC1, JM1, KC2, KC3), which are widely used benchmarks for software fault prediction.  
Each dataset contains software metrics (e.g., lines of code, cyclomatic complexity) and a binary defect label.

- [PROMISE Dataset Repository (GitHub)](https://github.com/ApoorvaKrisna/NASA-promise-dataset-repository)
- [PROMISE Dataset (Figshare)](https://figshare.com/articles/dataset/Software_Defect_Prediction_Dataset/13536506)

**Note:** Place the dataset (e.g., `pc1.csv`) in the `data/` directory.

---

## Installation

1. **Clone the repository:**
git clone https://github.com/HannaNajih/Classification_Lago_implementation_hybrid_approuch.git
cd Classification_Lago_implementation_hybrid_approuch

jupyter notebook notebooks/fault_prediction.ipynb
Follow the notebook cells to:
- Load and preprocess the data
- Train and evaluate classical classifiers
- Compare results with those reported in the literature

3. **Results:**  
Evaluation metrics (accuracy, precision, recall, F1-score) are saved in `results/classification_results.csv` and displayed in the notebook.

---

## Experiments & Results

- The notebook implements:
 - **Decision Tree (CART)**
 - **k-Nearest Neighbors**
 - **Support Vector Machine**
 - **Random Forest**
- Evaluation is performed using **10-fold cross-validation**.
- Results are compared with those reported in the reference paper for the same dataset.

**Sample Results Table:**

| Classifier           | Accuracy | Precision | Recall | F1-score |
|----------------------|----------|-----------|--------|----------|
| Decision Tree        | 0.85     | 0.27      | 0.21   | 0.21     |
| K-Nearest Neighbor   | 0.89     | 0.20      | 0.04   | 0.07     |
| Support Vector Machine | 0.90   | 0.00      | 0.00   | 0.00     |
| Random Forest        | 0.89     | 0.05      | 0.02   | 0.03     |

---

## Comparison with Literature

Results are directly compared with the metrics reported in:

> **A Hybrid Approach Based on k-Nearest Neighbors and Decision Tree for Software Fault Prediction**  
> [S. K. Dubey, A. Rana, et al.]

| Classifier           | Accuracy (Our) | Accuracy (Paper) | Precision (Our) | Precision (Paper) | Recall (Our) | Recall (Paper) | F1-score (Our) | F1-score (Paper) |
|----------------------|----------------|------------------|-----------------|-------------------|--------------|----------------|----------------|------------------|
| Decision Tree        | 0.85           | 0.91             | 0.27            | 0.58              | 0.21         | 0.51           | 0.21           | 0.55             |
| K-Nearest Neighbor   | 0.89           | 0.92             | 0.20            | 0.62              | 0.04         | 0.56           | 0.07           | 0.59             |
| Support Vector Machine | 0.90         | 0.92             | 0.00            | 0.64              | 0.00         | 0.57           | 0.00           | 0.60             |
| Random Forest        | 0.89           | 0.93             | 0.05            | 0.65              | 0.02         | 0.60           | 0.03           | 0.62             |

**Discussion:**  
Our accuracy results are close to those in the literature, but precision, recall, and F1-score are lower, likely due to class imbalance and differences in preprocessing or parameter tuning. Future work may involve class balancing and hyperparameter optimization for improved minority class detection.

---

## References

- Dubey, S. K., Rana, A., et al. (2023). *A hybrid approach based on k-nearest neighbors and decision tree for software fault prediction*. [PDF Link](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/72383936/1e7e0baa-a9c4-4618-a49c-ea1cecfe0069/18331-Final-MS-116953-1-10-20230310-1.pdf)
- [PROMISE Repository](http://promise.site.uottawa.ca/SERepository/datasets-page.html)
- [NASA PROMISE Datasets (GitHub)](https://github.com/ApoorvaKrisna/NASA-promise-dataset-repository)

---

## License

This project is for academic and research purposes. Refer to the repository license for details.

---

*For questions or contributions, please open an issue or pull request.*



