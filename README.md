# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a customer is likely to churn based on demographic and usage-related features.

## Project Overview
Customer churn is a critical business problem where organizations aim to identify customers who are likely to discontinue their services. This project builds a supervised machine learning pipeline to predict churn and deploys the model using a Streamlit web application.

## Dataset
The dataset contains customer-level information including:
- Age
- Gender
- Tenure
- Monthly Charges
- Churn (Target Variable)

## Exploratory Data Analysis
- Analyzed churn distribution
- Checked for missing values and duplicates
- Studied relationships between churn, age, tenure, gender, and monthly charges
- Visualized churn patterns using pie charts, bar charts, and histograms

## Feature Engineering
- Converted categorical variables into numeric format
- Encoded target variable (Churn: Yes/No → 1/0)
- Selected relevant features based on correlation and domain understanding
- Standardized numerical features using StandardScaler

## Model Building
Multiple classification models were trained and evaluated:
- Logistic Regression
- K-Nearest Neighbors (with GridSearchCV)
- Support Vector Machine (with GridSearchCV)
- Decision Tree (with GridSearchCV)
- Random Forest (with GridSearchCV)

Hyperparameter tuning was performed using 5-fold cross-validation.

## Model Selection
The Support Vector Machine (SVM) model achieved the best overall performance and was selected as the final model.

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
