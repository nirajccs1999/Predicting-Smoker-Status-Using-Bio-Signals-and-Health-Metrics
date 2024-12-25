# Predicting-Smoker-Status-Using-Bio-Signals-and-Health-Metrics

This project uses multiple machine learning algorithms to predict whether an individual smokes or not based on various health-related features. The models are evaluated using performance metrics like accuracy, AUC-ROC, and cross-validation scores.

## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Algorithms Used](#algorithms-used)
- [Model Evaluation](#model-evaluation)
- [Files](#files)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)

## Project Description

The goal of this project is to develop a machine learning model that predicts smoking behavior based on health and demographic data. The dataset contains various health-related features such as height, weight, and hemoglobin levels.

The project compares the performance of multiple machine learning algorithms including **Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost** to assess which model provides the best prediction accuracy.

## Dataset

The dataset used in this project consists of various health-related features that can be used to predict smoking behavior. The dataset has two primary classes:

- **Class 0**: Non-Smoking
- **Class 1**: Smoking

### Key Features:
- **Height**: Height of the individual (in cm).
- **Hemoglobin**: Hemoglobin levels in the blood.
- **Other Health Parameters**: Various health-related features that may influence smoking behavior.

## Algorithms Used

### 1. **Logistic Regression**
- **Accuracy**: 0.75
- **AUC-ROC**: 0.83
- **Cross-validation Accuracy**: 0.75

### 2. **Decision Tree Classifier**
- **Accuracy**: 0.75
- **AUC-ROC**: 0.83
- **Cross-validation Accuracy**: 0.75

#### Model Hyperparameters:
- `max_depth=6`

### 3. **Random Forest Classifier**
- **Accuracy**: 0.76
- **Cross-validation Accuracy**: 0.76

#### Model Hyperparameters:
- `max_depth=6`
- `random_state=42`

### 4. **XGBoost Classifier**
- **Accuracy**: 0.77
- **Cross-validation Accuracy**: 0.78

#### Model Hyperparameters:
- `max_depth=10`
- `learning_rate=0.05`
- `n_estimators=150`
- `eval_metric='logloss'`
- `random_state=42`

## Model Evaluation

### Metrics Used:
- **Accuracy**: Proportion of correctly classified instances.
- **AUC-ROC**: Area Under the Curve of the Receiver Operating Characteristic, used to evaluate the binary classification performance.
- **Cross-validation Accuracy**: Accuracy scores calculated using 10-fold cross-validation to assess the model’s generalization ability.

## Files

- **`smoke.ipynb`**: This is the Jupyter notebook where the code for data preprocessing, model training, and evaluation is written. It contains the implementation of the machine learning algorithms and performance evaluation.
- **`smoke_dataset.csv`**: The main dataset used for training and testing the models. This dataset contains various health-related features to predict smoking behavior.
- **`smoke_dataset-test.csv`**: This is the unseen data used for making predictions using the trained model.
- **`smoke_dataset_test_with_prediction.csv`**: This file contains the predictions made by the model on the `smoke_dataset-test.csv` file. It includes the original data along with the predicted smoking status.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/nirajccs1999/Predicting-Smoker-Status-Using-Bio-Signals-and-Health-Metrics.git

2. Navigate to the project directory:
3. Install the required libraries:


## Results

### Logistic Regression:

    Accuracy: 0.75
    AUC-ROC: 0.83
    Cross-validation Accuracy: 0.75

### Decision Tree Classifier:

    Accuracy: 0.75
    AUC-ROC: 0.83
    Cross-validation Accuracy: 0.75

### Random Forest Classifier:

    Accuracy: 0.76
    AUC-ROC: Not specified
    Cross-validation Accuracy: 0.76


### XGBoost Classifier:

    Accuracy: 0.77
    AUC-ROC: Not specified
    Cross-validation Accuracy: 0.78

## Conclusion

After testing multiple machine learning models, XGBoost emerged as the top-performing model with the highest accuracy of 77% on test data and consistently strong performance across cross-validation (average CV score: ~78%). Logistic Regression and Decision Tree Classifier followed, each achieving 75% accuracy with average CV scores of ~75%. The Random Forest Classifier also performed decently with 76% accuracy and a CV score of ~76%.

