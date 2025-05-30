# Logistic Regression-Based Classification for Breast Cancer Diagnosis

-----

## Overview

This project implements a **binary classification model** using **Logistic Regression** to diagnose breast cancer based on the comprehensive Wisconsin Breast Cancer Dataset. The goal is to build a robust and interpretable model that can accurately distinguish between **benign (non-cancerous)** and **malignant (cancerous)** tumors.

The repository covers the entire machine learning pipeline, from data loading and preprocessing to model training, evaluation, and in-depth analysis of key metrics like precision, recall, and ROC-AUC. It also includes visualizations to better understand the data and the model's performance.

-----

## Features

  * **Data Loading & Exploration**: Imports and initial analysis of the Breast Cancer Wisconsin dataset.
  * **Data Preprocessing**: Handles data cleaning (dropping irrelevant columns) and feature scaling using `StandardScaler`.
  * **Train-Test Split**: Splits the dataset into training and testing sets to ensure robust model evaluation.
  * **Logistic Regression Model**: Implements and trains a Logistic Regression classifier.
  * **Model Evaluation**: Assesses model performance using:
      * **Confusion Matrix**
      * **Accuracy, Precision, Recall, F1-Score**
      * **ROC Curve and AUC (Area Under the Curve)**
  * **Threshold Tuning**: Demonstrates how to adjust the classification threshold to optimize for specific metrics (e.g., higher recall to minimize false negatives in medical diagnosis).
  * **Sigmoid Function Explanation**: Provides a clear explanation of the underlying sigmoid function in logistic regression.
  * **Visualizations**: Includes various plots for:
      * Feature distributions for different diagnoses.
      * Correlation heatmap of features.
      * Pairplots of key features.
      * Logistic Regression coefficients.
      * Decision boundary visualization (using PCA for 2D representation).
      * Predicted probability distributions for each class.

-----

## Dataset

The project utilizes the **Breast Cancer Wisconsin (Diagnostic) Dataset**.

This dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

**Features included:**

  * `id` (dropped)
  * `diagnosis` (Target: 'M' = Malignant, 'B' = Benign)
  * `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`, `concave points_mean`, `symmetry_mean`, `fractal_dimension_mean`
  * `radius_se`, `texture_se`, `perimeter_se`, `area_se`, `smoothness_se`, `compactness_se`, `concavity_se`, `concave points_se`, `symmetry_se`, `fractal_dimension_se`
  * `radius_worst`, `texture_worst`, `perimeter_worst`, `area_worst`, `smoothness_worst`, `compactness_worst`, `concavity_worst`, `concave points_worst`, `symmetry_worst`, `fractal_dimension_worst`

-----

## How to Run

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/SiddardhaShayini/Logistic-Regression-Based-Classification-for-Breast-Cancer-Diagnosis-in-Wisconsin-Dataset.git
    cd Logistic-Regression-Based-Classification-for-Breast-Cancer-Diagnosis-in-Wisconsin-Dataset
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**

    ```bash
    pip install pandas scikit-learn matplotlib seaborn numpy
    ```

4.  **Open and run the Jupyter Notebook:**

    ```bash
    jupyter notebook Breast_Cancer_Diagnosis.ipynb
    ```

    Follow the steps in the notebook to execute the code, train the model, and view the results and visualizations.

-----

## Results & Insights

The Logistic Regression model achieved excellent performance on this dataset. Key metrics from the analysis include:

  * **Accuracy: 0.9737** (Almost 97.4% of cases correctly classified)
  * **Precision: 0.9756** (Very low false positive rate, minimizing incorrect malignant diagnoses)
  * **Recall: 0.9524** (Successfully identified over 95% of actual malignant cases, crucial for medical diagnosis)
  * **F1-Score: 0.9639** (Strong balance between precision and recall)
  * **ROC-AUC Score: 0.9960** (Exceptional discriminative power, indicating the model is highly effective at distinguishing between benign and malignant cases)

These results highlight the effectiveness of Logistic Regression for this binary classification task, especially after proper data preprocessing. The ability to tune the classification threshold allows for adapting the model to prioritize either precision or recall, depending on the specific clinical requirements.

-----

## ✍️ Author

- Siddardha Shayini

  
-----
