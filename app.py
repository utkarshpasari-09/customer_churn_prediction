# Gender 1 Female  0 Male
# Churn  1 Yes  0 No
# model is exported as model.pkl
# order of the X 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fb;
    }
    .block-container {
        background: #ffffff;
        padding: 2.5rem;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        max-width: 900px;
    }
    .result-high {
        background: #fee2e2;
        border-left: 6px solid #dc2626;
        padding: 1.2rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .result-low {
        background: #ecfdf5;
        border-left: 6px solid #16a34a;
        padding: 1.2rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")
    return scaler, model

scaler, model = load_artifacts()

st.title("Customer Churn Prediction")
st.caption("ML-based risk assessment for customer churn")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=130, value=10)
    monthly_charge = st.number_input(
        "Monthly Charges", min_value=30.0, max_value=150.0, value=70.0
    )

gender_encoded = 1 if gender == "Female" else 0
input_data = np.array([[age, gender_encoded, tenure, monthly_charge]])

st.markdown("---")

predict = st.button("Predict Churn Risk", use_container_width=True)

if predict:
    X_scaled = scaler.transform(input_data)
    prediction = model.predict(X_scaled)[0]

    st.markdown("### Prediction Result")

    if prediction == 1:
        st.markdown(
            """
            <div class="result-high">
                <h4>🔴 High Risk of Churn</h4>
                <p>This customer shows strong indicators of churn.</p>
                <b>Recommended Action:</b> Offer retention incentives or proactive support.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="result-low">
                <h4>🟢 Low Risk of Churn</h4>
                <p>This customer is likely to continue using the service.</p>
                <b>Recommended Action:</b> Maintain engagement and service quality.
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("Built with Python · Scikit-learn · Streamlit")
