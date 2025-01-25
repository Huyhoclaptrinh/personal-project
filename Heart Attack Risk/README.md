
# Heart Attack Risk Classification

This project demonstrates a machine learning pipeline for classifying heart attack risk into three categories: **Low**, **Moderate**, and **High**. The implementation uses PyTorch for building and training a neural network model, along with exploratory data analysis and preprocessing using Pandas, NumPy, and Scikit-learn.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Steps Involved](#steps-involved)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Usage](#usage)
- [References](#references)

---

## Overview
Heart disease remains one of the leading causes of mortality worldwide. Early detection and risk stratification can help in reducing the burden of the disease. This project classifies individuals into **Low**, **Moderate**, or **High** risk of heart attack based on their medical and lifestyle features.

The pipeline includes:
- Data Preprocessing
- Exploratory Data Analysis
- Categorical Encoding
- Feature Scaling
- Neural Network Model Training

---

## Dataset
The dataset used in this project contains medical and lifestyle attributes associated with heart attack risk. The target variable, `Heart_Attack_Risk`, has three classes:
- `Low`
- `Moderate`
- `High`

The dataset contains:
- Categorical and numerical features.
- Preprocessed labels for classification.

The dataset file is named: **`heart_attack_risk_dataset.csv`**.

---

## Prerequisites
To run this project, you need to have the following installed:
- Python 3.8 or later
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Install the required Python packages using:
```bash
pip install -r requirements.txt
```

---

## Project Structure
```
Heart_Attack_Classification/
├── heart_attack_risk_dataset.csv    # Dataset file
├── Heart_Attack.ipynb              # Main Jupyter notebook for implementation
├── README.md                       # Project documentation
├── requirements.txt                # List of dependencies
```

---

## Steps Involved
### 1. Data Exploration
- **Class Distribution**: Visualized using a count plot to understand the balance of the target variable.
- **Correlation Matrix**: Generated for the first 10 features after one-hot encoding.

### 2. Preprocessing
- **Target Encoding**: Converted `Heart_Attack_Risk` values to numerical format using `LabelEncoder`.
- **Feature Encoding**: Applied one-hot encoding to categorical features.
- **Feature Scaling**: Standardized features using `StandardScaler`.

### 3. Splitting Dataset
- Split data into training and testing sets with an 80-20 ratio.

### 4. Dataset Preparation for PyTorch
- Created a custom dataset class `HeartAttackDataset` for PyTorch.
- Converted features and labels into tensors for model training.

### 5. Model Training
- Implemented a neural network using PyTorch with the following architecture:
  - Input layer
  - Two hidden layers (128 and 64 neurons)
  - Dropout (30%) after each hidden layer
  - Output layer with 3 logits (for Low, Moderate, and High)
- Trained the model using CrossEntropyLoss and Adam optimizer over 50 epochs.

### 6. Evaluation
- Evaluated the model on the test set.
- Visualized the confusion matrix.
- Generated a classification report with precision, recall, and F1-score.

---

## Model Architecture
```
HeartAttackNN(
  (fc1): Linear(in_features=..., out_features=128)
  (fc2): Linear(in_features=128, out_features=64)
  (fc3): Linear(in_features=64, out_features=3)
  (relu): ReLU()
  (dropout): Dropout(p=0.3, inplace=False)
)
```
The model includes two fully connected hidden layers with ReLU activation and dropout for regularization.

---

## Results
### Confusion Matrix
The confusion matrix shows the model's performance across all three classes:

- **Low**: True positives vs false positives and negatives.
- **Moderate**: Misclassifications between adjacent risk levels.
- **High**: Precision and recall for severe cases.

### Classification Report
- **Accuracy**: Overall model accuracy on the test set.
- **Precision, Recall, and F1-Score**: Evaluated for each class.

---

## Usage
1. Clone this repository:
   ```bash
   git clone <repository_url>
   ```

2. Navigate to the project directory:
   ```bash
   cd Heart_Attack_Classification
   ```

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook Heart_Attack.ipynb
   ```

---

## References
- PyTorch Documentation: https://pytorch.org
- Seaborn Documentation: https://seaborn.pydata.org
- Scikit-learn Documentation: https://scikit-learn.org
- Dataset: User-provided file

Feel free to reach out for questions or collaboration!
