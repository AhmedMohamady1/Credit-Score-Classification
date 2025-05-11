import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
from preprocessing import *

# Load the trained pipeline
@st.cache_resource
def load_model():
    with open('svc.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# App title and description
st.title("Credit Score Prediction System")
st.write("""
This app predicts the Credit Score (Poor, Standard, Good) based on customer financial information.
You can either input data manually or upload a CSV file for batch predictions.
""")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    # Sidebar for user input features
    st.sidebar.header('Customer Information')

    def user_input_features():
        # Basic info
        name = st.sidebar.text_input("Name", "John Doe")
        age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
        month = st.sidebar.selectbox("Month", [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])

        # Occupation and income
        occupation = st.sidebar.selectbox("Occupation", [
            "Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer", 
            "Lawyer", "Media_Manager", "Doctor", "Journalist", "Manager", 
            "Accountant", "Musician", "Mechanic", "Writer"
        ])
        annual_income = st.sidebar.number_input("Annual Income", min_value=0.0, value=50000.0)
        monthly_inhand_salary = st.sidebar.number_input("Monthly Inhand Salary", min_value=0.0, value=3000.0)

        # Financial accounts
        num_bank_accounts = st.sidebar.number_input("Number of Bank Accounts", min_value=0, value=3)
        num_credit_card = st.sidebar.number_input("Number of Credit Cards", min_value=0, value=4)
        interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=15.0)
        num_of_loan = st.sidebar.number_input("Number of Loans", min_value=0, value=2)

        # Loans
        type_of_loan = st.sidebar.text_input("Type of Loan", "Auto Loan, Personal Loan")

        # Payment history
        delay_from_due_date = st.sidebar.number_input("Average Delay from Due Date (days)", min_value=0, value=5)
        num_of_delayed_payment = st.sidebar.number_input("Number of Delayed Payments", min_value=0, value=5)
        changed_credit_limit = st.sidebar.number_input("Changed Credit Limit (%)", min_value=0.0, max_value=100.0, value=10.0)
        num_credit_inquiries = st.sidebar.number_input("Number of Credit Inquiries", min_value=0, value=3)
        credit_mix = st.sidebar.selectbox("Credit Mix", ["Bad", "Standard", "Good", "_"])

        # Debt info
        outstanding_debt = st.sidebar.number_input("Outstanding Debt", min_value=0.0, value=1000.0)
        credit_utilization_ratio = st.sidebar.number_input("Credit Utilization Ratio", min_value=0.0, max_value=100.0, value=30.0)
        credit_history_age = st.sidebar.text_input("Credit History Age (e.g., '5 Years and 3 Months')", "5 Years and 3 Months")

        # Payments
        payment_of_min_amount = st.sidebar.selectbox("Payment of Minimum Amount", ["No", "Yes", "NM"])
        total_emi_per_month = st.sidebar.number_input("Total EMI per Month", min_value=0.0, value=300.0)
        amount_invested_monthly = st.sidebar.number_input("Amount Invested Monthly", min_value=0.0, value=200.0)
        payment_behaviour = st.sidebar.selectbox("Payment Behaviour", [
            "Low_spent_Small_value_payments", "High_spent_Medium_value_payments",
            "Low_spent_Medium_value_payments", "High_spent_Large_value_payments",
            "High_spent_Small_value_payments", "Low_spent_Large_value_payments"
        ])
        monthly_balance = st.sidebar.number_input("Monthly Balance", value=1000.0)

        # Create a dictionary
        data = {
            'Month': month,
            'Name': name,
            'Age': age,
            'Occupation': occupation,
            'Annual_Income': annual_income,
            'Monthly_Inhand_Salary': monthly_inhand_salary,
            'Num_Bank_Accounts': num_bank_accounts,
            'Num_Credit_Card': num_credit_card,
            'Interest_Rate': interest_rate,
            'Num_of_Loan': num_of_loan,
            'Type_of_Loan': type_of_loan,
            'Delay_from_due_date': delay_from_due_date,
            'Num_of_Delayed_Payment': num_of_delayed_payment,
            'Changed_Credit_Limit': changed_credit_limit,
            'Num_Credit_Inquiries': num_credit_inquiries,
            'Credit_Mix': credit_mix,
            'Outstanding_Debt': outstanding_debt,
            'Credit_Utilization_Ratio': credit_utilization_ratio,
            'Credit_History_Age': credit_history_age,
            'Payment_of_Min_Amount': payment_of_min_amount,
            'Total_EMI_per_month': total_emi_per_month,
            'Amount_invested_monthly': amount_invested_monthly,
            'Payment_Behaviour': payment_behaviour,
            'Monthly_Balance': monthly_balance
        }

        # Convert to DataFrame
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Display user input features
    st.subheader('Customer Input Features')
    st.write(input_df)

    # Make prediction
    if st.button('Predict Credit Score'):
        # Process the input through the pipeline
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df) if hasattr(model, 'predict_proba') else None
        
        # Display results
        st.subheader('Prediction')
        credit_score = prediction[0]
        
        # Custom styling for the prediction result
        if credit_score == 'Good':
            st.success(f'Predicted Credit Score: {credit_score} 👍')
        elif credit_score == 'Standard':
            st.warning(f'Predicted Credit Score: {credit_score} ➖')
        else:
            st.error(f'Predicted Credit Score: {credit_score} 👎')
        
        # Show prediction probabilities if available
        if prediction_proba is not None:
            st.subheader('Prediction Probability')
            proba_df = pd.DataFrame({
                'Class': model.classes_,
                'Probability': prediction_proba[0]
            })
            st.bar_chart(proba_df.set_index('Class'))

with tab2:
    st.header("Batch Prediction")
    st.write("Upload a CSV file containing customer data for batch predictions.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            batch_df = pd.read_csv(uploaded_file)
            
            # Display the uploaded data
            st.subheader("Uploaded Data Preview")
            st.write(batch_df.head())
            
            if st.button('Predict Batch'):
                # Make predictions
                with st.spinner('Making predictions...'):
                    predictions = model.predict(batch_df)
                    
                    # Add predictions to the dataframe
                    result_df = batch_df.copy()
                    result_df['Predicted_Credit_Score'] = predictions
                    
                    # Display results
                    st.subheader("Prediction Results")
                    st.write(result_df)
                    
                    # Download button
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name='credit_score_predictions.csv',
                        mime='text/csv'
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Add some explanations
st.subheader('How to interpret the results')
st.markdown("""
- **Good**: The customer has excellent creditworthiness
- **Standard**: The customer has average creditworthiness
- **Poor**: The customer has poor creditworthiness and may be high risk
""")

# Add some tips for improving credit score
st.subheader('Tips to Improve Credit Score')
st.markdown("""
1. Pay your bills on time
2. Keep credit utilization below 30%
3. Don't apply for too much credit at once
4. Maintain a healthy mix of credit types
5. Regularly check your credit report for errors
""")