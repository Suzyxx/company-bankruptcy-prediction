# Corporate Bankruptcy Prediction

## Overview
This project predicts corporate bankruptcy using a high-dimensional and imbalanced financial dataset containing 95 financial indicators. The goal is to evaluate and compare multiple machine learning models while analyzing their behavior using appropriate metrics for imbalanced classification.

---

## Problem
- **Task**: Binary classification (Bankrupt vs Non-bankrupt)
- **Key challenges**:
  - High dimensionality
  - Class imbalance
  - Costly false negatives (missed bankruptcies)

---

## Methodology
- Exploratory data analysis and preprocessing
- Stratified train/test split
- Feature selection embedded within cross-validation to prevent data leakage
- Model pipelines with imbalance-aware configurations

### Models evaluated
- Logistic Regression  
- Random Forest  
- XGBoost (with hyperparameter tuning)  
- Neural Network (MLP)

---

## Evaluation
- **Cross-validation**: Stratified 5-fold
- **Primary metric**: ROC-AUC
- **Secondary metric**: F1-score
- **Final evaluation**: Held-out test set with confusion matrix analysis

---

## Results (Test Set)

| Model               | ROC-AUC | F1 |
|--------------------|--------:|---:|
| Logistic Regression | 0.916 | 0.30 |
| Random Forest       | 0.966 | 0.48 |
| XGBoost             | 0.950 | 0.47 |
| XGBoost (Tuned)     | 0.954 | 0.51 |
| Neural Network      | 0.837 | 0.27 |

---

## Key Takeaways
- Tree-based ensemble models performed best on this tabular financial dataset.
- High ROC-AUC does not guarantee high F1 score in imbalanced problems.
- Model selection should consider real-world cost trade-offs, not only accuracy.

---

## Tools
Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn

---

## Project Context
This project was completed as an **independent machine learning assignment**.  
All analysis, modeling, evaluation, and interpretation were conducted individually.
