import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Fraud Detection System",page_icon="💳",layout="centered")
x = pd.read_csv("C:\\Users\\HP\\x_train_fraud_detect.csv")
feature_columns = x.columns.tolist()

# GET FRAUD DETECTION OPTIONS
location_options = x.columns[5:14].tolist()
device_options = x.columns[14:18].tolist()
payment_options = x.columns[18:24].tolist()
transaction_options = x.columns[24:29].tolist()
model = joblib.load("C:\\Users\\HP\\Downloads\\model_fraud_detection.joblib")
st.title("💳 Fraud Detection System")
st.write(
    "Enter the transaction information "
    "below to detect fraudulent transactions."
)
st.divider()
st.subheader("Enter Transaction Information")

transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0,step=10.0)

time = st.number_input("Time of Transaction",min_value=0.0,max_value=23.0,value=12.0,step=1.0)

previous_fraud = st.number_input("Previous Fraudulent Transactions",min_value=0,value=0,step=1)
account_age = st.number_input( "Account Age",min_value=0,value=30,step=1)

transactions_24h = st.number_input("Number of Transactions Last 24H",min_value=0,value=5,step=1)

location = st.selectbox("Transaction Location",location_options)
device = st.selectbox("Device",device_options)
# PAYMENT METHOD
payment_method = st.selectbox( "Payment Method",payment_options)

# TRANSACTION TYPE
transaction_type = st.selectbox("Transaction Type",transaction_options)

st.divider()

# PREDICTION
if st.button("🔍 Detect Fraud", use_container_width=True):
    input_data = pd.DataFrame(
        0.0,
        index=[0],
        columns=feature_columns
    )
    # INSERT NUMERICAL FEATURES

    input_data.iloc[0, 0] = transaction_amount
    input_data.iloc[0, 1] = time
    input_data.iloc[0, 2] = previous_fraud
    input_data.iloc[0, 3] = account_age
    input_data.iloc[0, 4] = transactions_24h

    # INSERT CATEGORICAL FEATURES
    input_data.loc[0, location] = 1
    input_data.loc[0, device] = 1
    input_data.loc[0, payment_method] = 1
    input_data.loc[0, transaction_type] = 1

    # PREDICTION
    prediction = model.predict(input_data)
    predicted_result = prediction[0]

    # DISPLAY RESULT
    st.success("Fraud detection completed successfully!")
    st.subheader("Fraud Detection Result")

    if predicted_result == 1:
        st.error( "⚠️ Fraudulent Transaction")

    else:
        st.success( "✅ Legitimate Transaction")
    # PREDICTION DETAILS
    st.subheader("Transaction Details")
    result = pd.DataFrame({
        "Feature": [
            "Transaction Amount",
            "Time of Transaction",
            "Previous Fraudulent Transactions",
            "Account Age",
            "Transactions Last 24H",
            "Location",
            "Device",
            "Payment Method",
            "Transaction Type"
        ],

        "Value": [
            f"{transaction_amount:,.2f}",
            f"{time:.0f}:00",
            previous_fraud,
            account_age,
            transactions_24h,
            location,
            device,
            payment_method,
            transaction_type
        ]
   })
    st.table(result)
