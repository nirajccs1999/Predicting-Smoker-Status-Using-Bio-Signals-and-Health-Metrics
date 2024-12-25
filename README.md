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
#### Data Description  

### Bio-Signals and Health Parameters  
In this project, we are using various bio-signals and health parameters to predict smoker status. Here's a breakdown of what we mean by **Bio-Signals** and **Health Parameters**:  

#### Bio-Signals  
Bio-signals are physiological readings or markers that reflect the state of the body. They provide insights into an individual's health condition. In our dataset, the following columns can be considered bio-signals:  

- **Eyesight (left):** A measure of visual ability in the left eye.  
- **Eyesight (right):** A measure of visual ability in the right eye.  
- **Hearing (left):** A measure of hearing ability in the left ear.  
- **Hearing (right):** A measure of hearing ability in the right ear.  
- **Hemoglobin:** A protein found in red blood cells that carries oxygen. Abnormal levels may indicate certain health conditions.  
- **Serum Creatinine:** A marker used to evaluate kidney function.  
- **AST (Aspartate Aminotransferase):** An enzyme related to liver function.  
- **ALT (Alanine Aminotransferase):** Another enzyme related to liver function.  
- **GTP (Gamma-glutamyl Transferase):** An enzyme often linked with liver health.  

#### Health Parameters  
Health parameters are broader metrics that relate to an individual’s overall health, lifestyle, or risk factors. These indicators help gauge an individual's well-being and the risk factors associated with diseases or conditions. In the dataset, these columns represent health parameters:  

- **Age:** A general demographic parameter that provides context to the individual's health.  
- **Height (cm):** A general health indicator that relates to body size.  
- **Weight (kg):** Another general health indicator related to body weight.  
- **Waist (cm):** A measurement indicating body fat distribution, often used as an indicator of cardiovascular risk.  
- **Systolic:** The systolic reading in blood pressure, which measures the force of blood against artery walls.  
- **Relaxation:** A measure of stress level or relaxation, which can affect overall health.  
- **Fasting Blood Sugar:** Blood sugar levels after fasting, useful for identifying diabetes or pre-diabetes conditions.  
- **Cholesterol:** A key parameter for heart health, higher levels can be a risk factor for cardiovascular diseases.  
- **Triglyceride:** A type of fat in the blood, high levels are also linked with heart disease.  
- **HDL (High-Density Lipoprotein):** Often called "good cholesterol," higher levels can protect against heart disease.  
- **LDL (Low-Density Lipoprotein):** Known as "bad cholesterol," higher levels increase the risk of heart disease.  
- **Urine Protein:** Indicates kidney function or potential kidney damage.  
- **Dental Caries:** An indicator of oral health, which can impact overall health if untreated.  



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

