# Breast Cancer Classification App

A machine learning-powered web application for breast cancer prediction and diagnosis using comparative model benchmarking and advanced preprocessing pipelines. This project evaluates multiple machine learning algorithms to classify tumors as **Benign** or **Malignant**, followed by deployment using Streamlit for real-time inference.

---

# Project Overview

Breast cancer is among the leading causes of cancer-related deaths worldwide. Early detection significantly improves survival rates and treatment effectiveness.

This project aims to:

- Build an accurate breast cancer classification system
- Compare multiple ML algorithms systematically
- Analyze the effect of preprocessing pipelines
- Deploy the best-performing model as a web application

The project follows a complete end-to-end machine learning workflow including:

1. Data preprocessing  
2. Feature engineering  
3. Model training  
4. Benchmarking and evaluation  
5. Hyperparameter optimization  
6. Deployment using Streamlit  

---

# Demo

The application allows users to:

- Input diagnostic values manually
- Predict whether the tumor is malignant or benign
- View prediction confidence
- Interact with a clean Streamlit interface

---

# Dataset

The project uses the **Breast Cancer Wisconsin Diagnostic Dataset**.

## Dataset Information

The dataset contains features extracted from digitized images of fine needle aspirates (FNA) of breast masses.

### Examples of Features

| Feature | Description |
|---|---|
| Radius | Mean distance from center to perimeter |
| Texture | Standard deviation of gray-scale values |
| Perimeter | Tumor perimeter |
| Area | Tumor area |
| Smoothness | Local variation in radius lengths |
| Compactness | Perimeter² / area |
| Concavity | Severity of concave portions |
| Symmetry | Tumor symmetry |
| Fractal Dimension | Complexity measure |

---

# Problem Statement

Classify tumors into:

- **Malignant (Cancerous)**
- **Benign (Non-Cancerous)**

using supervised machine learning models trained on diagnostic features.

---

# Tech Stack

## Programming Language
- Python

## Libraries Used

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn
- XGBoost

### Deployment
- Streamlit

---

# Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Preprocessing Pipelines
     ↓
Feature Engineering
     ↓
Model Training
     ↓
Benchmarking
     ↓
Hyperparameter Tuning
     ↓
Best Model Selection
     ↓
Deployment using Streamlit
```

---

# Preprocessing Pipelines

Three preprocessing strategies were tested to evaluate model robustness and performance.

---

## Pipeline 1 — Standard Scaling Pipeline

### Steps
- Missing value handling
- Feature normalization using `StandardScaler`
- Train-test split

### Purpose
Standardization improves performance for algorithms sensitive to feature magnitude.

### Best Suited For
- Logistic Regression
- Support Vector Machine (SVM)
- KNN
- SGD Classifier

---

## Pipeline 2 — MinMax Scaling + Feature Selection

### Steps
- MinMax normalization
- Correlation analysis
- Removal of redundant features

### Purpose
- Reduce dimensionality
- Improve computational efficiency
- Remove multicollinearity

### Advantages
- Faster training
- Better generalization
- Reduced overfitting

---

## Pipeline 3 — PCA-Based Dimensionality Reduction

### Steps
- Principal Component Analysis (PCA)
- Variance retention analysis
- Dimensionality reduction

### Purpose
- Compress feature space
- Improve model generalization
- Reduce noise

### Advantages
- Better visualization
- Reduced computational complexity
- Lower overfitting risk

---

# Models Evaluated

A total of **14 machine learning models** were benchmarked.

---

# 1. Logistic Regression

## Description
A statistical linear classification algorithm used for binary classification.

## Why Used
- Fast and interpretable
- Works well on linearly separable data

## Advantages
- Simple implementation
- Good baseline model
- Probabilistic output

## Limitations
- Struggles with non-linear relationships

---

# 2. Decision Tree

## Description
A tree-based model that splits data recursively based on feature importance.

## Advantages
- Easy to interpret
- Handles non-linear data

## Limitations
- Prone to overfitting

---

# 3. Random Forest

## Description
An ensemble learning algorithm combining multiple decision trees.

## Why Used
- High accuracy
- Robust against overfitting

## Advantages
- Excellent generalization
- Handles feature interactions well

## Limitations
- Higher computational cost

---

# 4. Support Vector Machine (SVM)

## Description
Finds the optimal hyperplane maximizing class separation.

## Advantages
- Effective in high-dimensional spaces
- Strong classification performance

## Limitations
- Computationally expensive for large datasets

---

# 5. K-Nearest Neighbors (KNN)

## Description
Classifies based on proximity to neighboring points.

## Advantages
- Simple and intuitive

## Limitations
- Sensitive to scaling
- Slower during inference

---

# 6. Naive Bayes

## Description
Probabilistic classifier based on Bayes’ theorem.

## Advantages
- Fast training
- Works well on smaller datasets

## Limitations
- Assumes feature independence

---

# 7. Gradient Boosting

## Description
Sequential ensemble model improving weak learners iteratively.

## Advantages
- High predictive power

## Limitations
- Slower training

---

# 8. AdaBoost

## Description
Boosting algorithm focusing on correcting previous errors.

## Advantages
- Improves weak learners

## Limitations
- Sensitive to noisy data

---

# 9. XGBoost

## Description
Optimized gradient boosting framework.

## Why Used
- State-of-the-art performance
- Efficient handling of structured data

## Advantages
- High accuracy
- Built-in regularization
- Parallel processing support

## Limitations
- More hyperparameters to tune

---

# 10. Extra Trees Classifier

## Description
Ensemble method using randomized decision trees.

## Advantages
- Faster than Random Forest
- Lower variance

---

# 11. Bagging Classifier

## Description
Uses bootstrap aggregation to improve stability.

## Advantages
- Reduces variance
- Improves robustness

---

# 12. Ridge Classifier

## Description
Linear classifier with L2 regularization.

## Advantages
- Prevents overfitting
- Stable performance

---

# 13. SGD Classifier

## Description
Uses stochastic gradient descent for optimization.

## Advantages
- Efficient for large datasets
- Fast training

---

# 14. Voting Ensemble

## Description
Combines predictions from multiple models.

## Advantages
- Better overall robustness
- Improved accuracy

---

# Evaluation Metrics

Each model was evaluated using:

| Metric | Purpose |
|---|---|
| Accuracy | Overall correctness |
| Precision | Correct positive predictions |
| Recall | Ability to identify actual positives |
| F1-Score | Balance between precision and recall |
| ROC-AUC | Classification quality |
| Confusion Matrix | Prediction breakdown |

---

# Model Benchmarking

| Model | Accuracy | Strength |
|---|---|---|
| Random Forest | High | Strong generalization |
| XGBoost | Very High | Best boosting performance |
| SVM | High | Strong boundary separation |
| Logistic Regression | Moderate | Interpretable baseline |
| Voting Ensemble | Very High | Combined robustness |

> Replace with actual metrics from your experimentation notebook.

---

# Hyperparameter Tuning

Optimization techniques used:

- GridSearchCV
- Cross-validation
- Parameter tuning

### Parameters Tuned
- Tree depth
- Learning rate
- Number of estimators
- Regularization strength
- Kernel parameters

---

# Streamlit Deployment

The best-performing model was deployed using Streamlit.

## Features of the App

- Interactive user interface
- Real-time prediction
- Easy deployment
- Browser-based accessibility
- Lightweight architecture

---

# Project Structure

```text
breast-cancer-classification-app/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── experimentation.ipynb
│
├── models/
│   └── best_model.pkl
│
├── app.py
├── utils.py
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Ananyajainhdjsj/breast-cancer-classification-app.git
```

---

## Navigate to Folder

```bash
cd breast-cancer-classification-app
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

---

# Future Improvements

- Add deep learning models
- Integrate SHAP/LIME explainability
- Docker deployment
- Cloud hosting (AWS/GCP/Azure)
- API integration
- Model monitoring pipeline

---

# Key Learnings

This project helped in understanding:

- Comparative model benchmarking
- End-to-end ML pipelines
- Feature engineering
- Ensemble learning
- Model deployment
- Hyperparameter optimization
- Healthcare AI applications

---

# Conclusion

This project demonstrates the practical application of machine learning in healthcare diagnostics through breast cancer prediction. By benchmarking 14 machine learning models across multiple preprocessing pipelines, the project identifies robust classification approaches and deploys them in a real-world accessible application using Streamlit.

The project highlights both the analytical and deployment aspects of building production-ready AI systems.
