# Wine Quality Prediction Model

This project aims to predict the quality of wine based on various chemical properties using several machine learning algorithms. The best performing model is then used to create a GUI for easy prediction.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [GUI Application](#gui-application)

## Overview

The goal of this project is to classify wines into two categories: good quality and bad quality, based on their chemical properties. We used several machine learning algorithms to achieve this and selected the best-performing model to build a GUI application for wine quality prediction.

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/your-username/wine-quality-prediction.git
    cd wine-quality-prediction
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Ensure that the dataset `data.csv` is in the root directory of the project.
2. Run the Jupyter notebook `wine_quality_prediction.ipynb` to train and evaluate the models.
3. Save the best performing model.
4. Run the `wine_quality_gui.py` script to start the GUI application.

## Dataset

The dataset used in this project is a wine quality dataset that includes several chemical properties of wine and a quality rating. The dataset can be downloaded from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality).

## Preprocessing

1. **Handling Missing Values**: Check for and handle any missing values in the dataset.
2. **Binarization**: The quality scores are binarized into two categories: good (quality >= 7) and bad (quality < 7).
3. **Handling Imbalance**: Use SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.
4. **Feature Scaling**: Standardize the features using StandardScaler.
5. **PCA**: Apply Principal Component Analysis (PCA) to reduce dimensionality while retaining 90% variance.

## Model Training

The following machine learning models were trained:

- Logistic Regression
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier

## Evaluation

The models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score

The Random Forest Classifier achieved the highest accuracy and was chosen as the final model.

## GUI Application

A simple GUI application was created using Tkinter to input wine chemical properties and predict its quality. 

### To run the GUI:

1. Ensure the best model (`wine_quality_prediction`) is saved in the project directory.
2. Run the `wine_quality_gui.py` script:

    ```sh
    python wine_quality_gui.py
    ```

3. Enter the required features and click the predict button to see the result.

## Acknowledgements

- The dataset used in this project is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality).
- The project was developed using various Python libraries including Pandas, Scikit-learn, Seaborn, and Tkinter.
