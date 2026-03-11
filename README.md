# Customer Churn Prediction

### Project Overview
This project predicts whether a telecom customer is likely to churn based on their service usage and demographic details. The goal is to help businesses identify high-risk customers and take preventive actions.

### Problem Statement
Customer churn is a major challenge for telecom companies. Predicting churn helps companies retain customers and reduce revenue loss.

### Dataset
Telecom customer dataset containing features like:
- Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Payment Method
- Internet Service

### Project Workflow
1. Data Cleaning
2. Feature Engineering
3. Exploratory Data Analysis
4. Model Training
5. Model Evaluation
6. Streamlit Web App Deployment

### Models Used
- Logistic Regression
- Random Forest

### Results

| Model &  Accuracy 
  Logistic Regression - 0.80 
  Random Forest - 0.79 

Logistic Regression performed slightly better.

### Key Insights
- Month-to-month contracts have higher churn rates
- Higher monthly charges increase churn risk
- Long-term customers are more loyal

### Web Application
A Streamlit app is built where users can input customer details and predict churn probability.

### Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib

### Project Structure
customer-churn-prediction
│
├── app
│ └── streamlit_app.py
│
├── src
│ ├── preprocess.py
│ ├── train_model.py
│ ├── churn_model.pkl
│ └── model_columns.pkl
│
├── data
│ └── churn.csv
│
├── requirements.txt
└── README.md


### Future Improvements
- Add advanced models like XGBoost
- Deploy the application online
- Add real-time prediction API

