# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a customer is likely to churn based on demographic and usage-related features. The project covers data analysis, model training, evaluation, and deployment using a Streamlit web application.

## Project Overview

Customer churn is a critical business problem where organizations aim to identify customers who are likely to discontinue their services. This project builds a supervised machine learning pipeline to predict churn and deploys the final model as an interactive web application.

## Dataset

The dataset contains customer-level information including:
- Age
- Gender
- Tenure
- Monthly Charges
- Churn (Target Variable)

## Exploratory Data Analysis

- Checked for missing values and duplicate records
- Analyzed churn distribution
- Studied relationships between churn and key features
- Visualized churn patterns using pie charts, bar charts, and histograms

## Feature Engineering

- Converted categorical variables into numeric format
- Encoded target variable (Churn: Yes/No → 1/0)
- Selected relevant features based on domain understanding
- Standardized numerical features using StandardScaler

## Model Building

The following classification models were trained and evaluated:
- Logistic Regression
- K-Nearest Neighbors (with GridSearchCV)
- Support Vector Machine (with GridSearchCV)
- Decision Tree (with GridSearchCV)
- Random Forest (with GridSearchCV)

Hyperparameter tuning was performed using 5-fold cross-validation.

## Model Selection

The Support Vector Machine (SVM) achieved the best overall performance and was selected as the final model.

## Deployment

- The trained model and scaler were saved using Joblib
- A Streamlit web application was developed for real-time churn prediction
- Users can input customer details and receive churn risk predictions

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib
- Scikit-learn
- Streamlit

## How to Run the App

```bash
pip install -r requirements.txt
streamlit run app.py



