"""
Interactive Prediction Script for Project 2: Customer Churn Prediction
Implements Step 6 from course workflow specifications:
- Asks user interactive input questions (tenure, monthly charges, total charges, senior citizen status, contract type, internet service)
- Processes input through the trained scaler & model pipeline
- Outputs prediction: Whether customer is likely to CHURN or STAY
- Displays churn probability percentage & risk assessment level!
"""

import os
import pickle
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(__file__)
MODEL_PKL = os.path.join(BASE_DIR, "churn_model.pkl")

def load_churn_pipeline():
    if not os.path.exists(MODEL_PKL):
        print(f"Error: Model file '{MODEL_PKL}' not found. Please run train_model.py first.")
        return None
    with open(MODEL_PKL, 'rb') as f:
        return pickle.load(f)

def get_user_inputs():
    print("\n=======================================================")
    print("   🔮 TELCO CUSTOMER CHURN PREDICTION INTERACTIVE CLI   ")
    print("=======================================================")
    print("Please answer the following customer details:\n")
    
    try:
        tenure = float(input("1. Customer Tenure in months (e.g. 1 to 72): ").strip() or "12")
        monthly_charges = float(input("2. Monthly Charges in $ (e.g. 18.0 to 120.0): ").strip() or "70.0")
        total_charges = float(input("3. Total Charges in $ (e.g. 70.0 to 8000.0): ").strip() or "840.0")
        senior = int(input("4. Senior Citizen status (1 for Yes, 0 for No): ").strip() or "0")
        
        print("\nContract Options: [1] month-to-month  [2] one-year  [3] two-year")
        c_choice = input("5. Select Contract Type (1/2/3): ").strip()
        contract_map = {'1': 'month-to-month', '2': 'one-year', '3': 'two-year'}
        contract = contract_map.get(c_choice, 'month-to-month')
        
        print("\nInternet Options: [1] fiber optic  [2] dsl  [3] no")
        i_choice = input("6. Select Internet Service (1/2/3): ").strip()
        internet_map = {'1': 'fiber optic', '2': 'dsl', '3': 'no'}
        internet_service = internet_map.get(i_choice, 'fiber optic')
        
        gender = input("7. Gender (Male/Female) [Default: Male]: ").strip().capitalize() or "Male"
        partner = input("8. Partner (Yes/No) [Default: No]: ").strip().capitalize() or "No"
        dependents = input("9. Dependents (Yes/No) [Default: No]: ").strip().capitalize() or "No"
        phone_service = input("10. Phone Service (Yes/No) [Default: Yes]: ").strip().capitalize() or "Yes"
        paperless = input("11. Paperless Billing (Yes/No) [Default: Yes]: ").strip().capitalize() or "Yes"
        payment_method = "Electronic check"
        
        return {
            'SeniorCitizen': senior,
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'avg_monthly_per_tenure': total_charges / (tenure + 1),
            'is_high_charge': 1 if monthly_charges > 65.0 else 0,
            'gender': gender,
            'Partner': partner,
            'Dependents': dependents,
            'PhoneService': phone_service,
            'InternetService': internet_service,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment_method
        }
    except Exception as e:
        print(f"Invalid input encountered ({e}). Defaulting to standard sample.")
        return {
            'SeniorCitizen': 0, 'tenure': 6, 'MonthlyCharges': 85.0, 'TotalCharges': 510.0,
            'avg_monthly_per_tenure': 510.0 / 7, 'is_high_charge': 1, 'gender': 'Female',
            'Partner': 'No', 'Dependents': 'No', 'PhoneService': 'Yes',
            'InternetService': 'fiber optic', 'Contract': 'month-to-month',
            'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check'
        }

def predict_single_customer(input_data, pipeline):
    model = pipeline['model']
    scaler = pipeline['scaler']
    label_encoders = pipeline['label_encoders']
    feature_cols = pipeline['feature_cols']
    cat_cols = pipeline['cat_cols']
    
    df_sample = pd.DataFrame([input_data])
    
    # Encode categorical columns safely
    for col in cat_cols:
        le = label_encoders[col]
        val = str(df_sample.loc[0, col])
        if val in le.classes_:
            df_sample[col] = le.transform([val])[0]
        else:
            df_sample[col] = 0
            
    # Scale features
    X_sample = df_sample[feature_cols]
    X_scaled = scaler.transform(X_sample)
    
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    churn_prob = probabilities[1] * 100
    
    print("\n=======================================================")
    print("             📊 PREDICTION RESULT & ANALYSIS           ")
    print("=======================================================")
    if prediction == 1:
        print(f"🚨 RESULT: CUSTOMER LIKELY TO CHURN!")
        print(f"🔥 Churn Probability: {churn_prob:.2f}%")
        print(f"⚠️  Risk Level: HIGH RISK")
        print("💡 Recommendation: Offer contract discount or retention incentives immediately.")
    else:
        print(f"✅ RESULT: CUSTOMER LIKELY TO STAY!")
        print(f"🛡️  Stay Probability: {100 - churn_prob:.2f}% (Churn Prob: {churn_prob:.2f}%)")
        print(f"💚 Risk Level: LOW RISK")
        print("💡 Recommendation: Customer is satisfied. Maintain standard service relationship.")
    print("=======================================================\n")

def main():
    pipeline = load_churn_pipeline()
    if not pipeline:
        return
    user_data = get_user_inputs()
    predict_single_customer(user_data, pipeline)

if __name__ == "__main__":
    main()
