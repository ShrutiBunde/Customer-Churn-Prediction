import streamlit as st
import pandas as pd
import pickle

# load model
model = pickle.load(open("src/churn_model.pkl", "rb"))
model_columns = pickle.load(open("src/model_columns.pkl", "rb"))
st.title("Customer Churn Prediction")

st.write("Enter customer details to predict churn")

# user inputs
tenure = st.slider("Tenure Months", 0, 72, 12)

monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

# convert inputs to dataframe
data = {
    "Tenure Months": tenure,
    "Monthly Charges": monthly_charges,
    "Total Charges": total_charges,
}

input_df = pd.DataFrame([data])
input_df = pd.get_dummies(input_df)

input_df = input_df.reindex(columns=model_columns, fill_value=0)
# predict button
if st.button("Predict Churn"):

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    if prediction[0] == 1:
       st.error(f"Customer likely to churn ⚠️ (Probability: {probability[0][1]*100:.2f}%)")
    else:
       st.success(f"Customer likely to stay ✅ (Probability: {probability[0][0]*100:.2f}%)")


st.subheader("Top Features Affecting Churn")

st.image("feature_importance.png")

st.subheader("Business Insights")

st.write("""
• Month-to-month contract customers churn more.\n
• Customers with high monthly charges show higher churn risk.\n
• Long tenure customers are less likely to churn.
""")

