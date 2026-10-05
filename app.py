import streamlit as st
import pandas as pd
import numpy as np
import joblib
import textwrap
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1 {
    font-size: 42px !important;
    font-weight: 700 !important;
}

h2 {
    margin-top: 30px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #3b82f6;
    background-color: #151a23;
    margin-top: 20px;
}

.probability {
    font-size: 48px;
    font-weight: 700;
}

.risk {
    font-size: 22px;
    font-weight: 700;
    margin-top: 10px;
}

.summary-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #151a23;
    border: 1px solid #2b303b;
}

.small-label {
    color: #9ca3af;
    font-size: 14px;
}

.big-value {
    font-size: 22px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL FILES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

try:

    model = joblib.load(MODEL_DIR / "churn_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

except Exception as e:

    st.error("❌ Model files could not be loaded.")
    st.code(str(e))
    st.stop()


# =========================================================
# HEADER
# =========================================================

st.title("📊 Customer Churn Predictor")

st.write(
    "Predict the probability of customer churn using Machine Learning."
)


# =========================================================
# PROJECT INFORMATION
# =========================================================

with st.expander("🎯 What does this project do?"):

    st.write(
        """
        This system analyzes important customer information and predicts
        the probability that a customer may leave the service.

        The Machine Learning model used in this project is Logistic Regression.
        """
    )

    st.write("🤖 **Machine Learning Model:** Logistic Regression")


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )

with col2:

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=2000.0,
        value=70.0,
        step=1.0
    )

with col3:

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=100000.0,
        value=1000.0,
        step=10.0
    )


# =========================================================
# SERVICE INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">🛰️ Service Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

with col2:

    online_security = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

with col3:

    tech_support = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )


# =========================================================
# ACCOUNT INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">💳 Account Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

with col2:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )

with col3:

    paperless_billing = st.selectbox(
        "Paperless Billing",
        [
            "No",
            "Yes"
        ]
    )

with col4:

    dependents = st.selectbox(
        "Dependents",
        [
            "No",
            "Yes"
        ]
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("---")

predict_button = st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True,
    key="predict_customer_churn"
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # CREATE EMPTY DATAFRAME WITH ALL 30 MODEL FEATURES
    # -----------------------------------------------------

    customer = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )


    # -----------------------------------------------------
    # NUMERICAL FEATURES
    # -----------------------------------------------------

    if "SeniorCitizen" in customer.columns:
        customer.loc[0, "SeniorCitizen"] = 0

    if "tenure" in customer.columns:
        customer.loc[0, "tenure"] = tenure

    if "MonthlyCharges" in customer.columns:
        customer.loc[0, "MonthlyCharges"] = monthly_charges

    if "TotalCharges" in customer.columns:
        customer.loc[0, "TotalCharges"] = total_charges


    # -----------------------------------------------------
    # DEPENDENTS
    # -----------------------------------------------------

    if dependents == "Yes":
        if "Dependents_Yes" in customer.columns:
            customer.loc[0, "Dependents_Yes"] = 1


    # -----------------------------------------------------
    # INTERNET SERVICE
    # -----------------------------------------------------

    if internet_service == "Fiber optic":

        if "InternetService_Fiber optic" in customer.columns:
            customer.loc[0, "InternetService_Fiber optic"] = 1

    elif internet_service == "No":

        if "InternetService_No" in customer.columns:
            customer.loc[0, "InternetService_No"] = 1

    # DSL is the baseline category
    # Therefore all InternetService dummy columns remain 0


    # -----------------------------------------------------
    # ONLINE SECURITY
    # -----------------------------------------------------

    if online_security == "Yes":

        if "OnlineSecurity_Yes" in customer.columns:
            customer.loc[0, "OnlineSecurity_Yes"] = 1

    elif online_security == "No internet service":

        if "OnlineSecurity_No internet service" in customer.columns:
            customer.loc[0, "OnlineSecurity_No internet service"] = 1


    # -----------------------------------------------------
    # TECH SUPPORT
    # -----------------------------------------------------

    if tech_support == "Yes":

        if "TechSupport_Yes" in customer.columns:
            customer.loc[0, "TechSupport_Yes"] = 1

    elif tech_support == "No internet service":

        if "TechSupport_No internet service" in customer.columns:
            customer.loc[0, "TechSupport_No internet service"] = 1


    # -----------------------------------------------------
    # CONTRACT
    # -----------------------------------------------------

    if contract == "One year":

        if "Contract_One year" in customer.columns:
            customer.loc[0, "Contract_One year"] = 1

    elif contract == "Two year":

        if "Contract_Two year" in customer.columns:
            customer.loc[0, "Contract_Two year"] = 1

    # Month-to-month is baseline


    # -----------------------------------------------------
    # PAPERLESS BILLING
    # -----------------------------------------------------

    if paperless_billing == "Yes":

        if "PaperlessBilling_Yes" in customer.columns:
            customer.loc[0, "PaperlessBilling_Yes"] = 1


    # -----------------------------------------------------
    # PAYMENT METHOD
    # -----------------------------------------------------

    if payment_method == "Credit card (automatic)":

        if "PaymentMethod_Credit card (automatic)" in customer.columns:
            customer.loc[
                0,
                "PaymentMethod_Credit card (automatic)"
            ] = 1

    elif payment_method == "Electronic check":

        if "PaymentMethod_Electronic check" in customer.columns:
            customer.loc[
                0,
                "PaymentMethod_Electronic check"
            ] = 1

    elif payment_method == "Mailed check":

        if "PaymentMethod_Mailed check" in customer.columns:
            customer.loc[
                0,
                "PaymentMethod_Mailed check"
            ] = 1

    # Bank transfer automatic is baseline


    # =====================================================
    # SCALE DATA
    # =====================================================

    try:

        customer_scaled = scaler.transform(customer)

    except Exception as e:

        st.error("❌ Error while scaling customer data.")
        st.code(str(e))
        st.stop()


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    try:

        churn_probability = model.predict_proba(
            customer_scaled
        )[0][1]

        prediction = model.predict(
            customer_scaled
        )[0]

    except Exception as e:

        st.error("❌ Error while making prediction.")
        st.code(str(e))
        st.stop()


    probability_percent = churn_probability * 100
    stay_probability = (1 - churn_probability) * 100


    # =====================================================
    # RISK LEVEL
    # =====================================================

    if probability_percent < 30:

        risk_level = "LOW RISK"
        risk_message = "Customer is currently less likely to churn."

    elif probability_percent < 60:

        risk_level = "MEDIUM RISK"
        risk_message = "Customer has a moderate possibility of churn."

    else:

        risk_level = "HIGH RISK"
        risk_message = "Customer has a higher possibility of churn."


    # ============================================================
    # PREDICTION RESULT
    # ============================================================

    st.markdown("---")

    st.markdown("## 📊 Prediction Result")

    # Risk level
    if probability_percent < 30:
        risk_level = "LOW RISK"
        risk_message = "Customer is currently less likely to churn."

    elif probability_percent < 60:
        risk_level = "MEDIUM RISK"
        risk_message = "Customer has a moderate possibility of churn."

    else:
        risk_level = "HIGH RISK"
        risk_message = "Customer has a higher possibility of churn."

    # Display prediction result
    st.metric(
        label="Churn Probability",
        value=f"{probability_percent:.2f}%"
    )

    if probability_percent < 30:

        st.success(
            f"🟢 {risk_level} — {risk_message}"
        )

    elif probability_percent < 60:

        st.warning(
            f"🟡 {risk_level} — {risk_message}"
        )

    else:

        st.error(
            f"🔴 {risk_level} — {risk_message}"
        )

    # ============================================================
    # PROBABILITY GRAPH
    # ============================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📈 Churn Probability</div>',
        unsafe_allow_html=True
    )

    chart_data = pd.DataFrame({
        "Status": ["Churn", "Stay"],
        "Probability": [
            probability_percent,
            100 - probability_percent
        ]
    })

    st.bar_chart(
        chart_data.set_index("Status")
    )

    # ============================================================
    # CUSTOMER SUMMARY
    # ============================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📋 Customer Summary</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="small-label">Tenure</div>
                <div class="big-value">{tenure} months</div>
                <br>
                <div class="small-label">Monthly Charges</div>
                <div class="big-value">${monthly_charges:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="small-label">Total Charges</div>
                <div class="big-value">${total_charges:.2f}</div>
                <br>
                <div class="small-label">Contract</div>
                <div class="big-value">{contract}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="small-label">Internet Service</div>
                <div class="big-value">{internet_service}</div>
                <br>
                <div class="small-label">Payment Method</div>
                <div class="big-value">{payment_method}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ============================================================
    # MODEL OUTPUT
    # ============================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🤖 Model Output</div>',
        unsafe_allow_html=True
    )

    if prediction == 1:

        st.error(
            "⚠️ Model predicts that this customer is likely to churn."
        )

    else:

        st.success(
            "✅ Model predicts that this customer is likely to stay."
        )   