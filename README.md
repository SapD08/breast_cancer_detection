# Breast Cancer Detection

Binary classification of breast tumours as malignant or benign using the Wisconsin Breast Cancer dataset from scikit-learn (569 samples, 30 numeric features).

## Goal

Catch as many malignant cases as possible. Missing a cancer is costlier than a false alarm, so malignant recall is the primary metric.

In this dataset `0 = malignant` and `1 = benign`, so recall is computed with `pos_label=0`.

## Workflow

1. Stratified 80/20 train/test split (`random_state=42`).
2. Baseline with a majority-class dummy classifier (62.6% accuracy).
3. Compared Logistic Regression, SVM and Random Forest with 5-fold stratified cross-validation on the training set. Scaling is done inside a `Pipeline` to avoid data leakage.
4. Tuned Logistic Regression (`C`, `class_weight`) with `GridSearchCV`.
5. Diagnosed the tuned model with out-of-fold predictions and chose a decision threshold of 0.4 from training cross-validation.
6. Evaluated once on the held-out test set.

## Results

Cross-validation on the training set (malignant recall):

| Model | Recall |
|---|---|
| Logistic Regression | 0.965 ± 0.029 |
| Random Forest | 0.959 ± 0.030 |
| SVM | 0.953 ± 0.040 |
| Logistic Regression (tuned) | 0.976 |

Held-out test set (threshold 0.4):

|  | Predicted malignant | Predicted benign |
|---|---|---|
| **Actually malignant** | 41 | 1 |
| **Actually benign** | 5 | 67 |

- Malignant recall: 0.98
- Malignant precision: 0.89
- Accuracy: 0.95

The test set is small (114 samples, 42 malignant), so a single misclassification shifts the metrics by a couple of points.

## Run

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook main.ipynb
```

## Files

- `main.ipynb`: full analysis and modelling
- `breast_cancer_model.joblib`: saved model
