"""
Data Loading & Dataset Generation Script for Project 2: Customer Churn Prediction
Generates a realistic Telco Customer Churn dataset incorporating all real-world features:
- Demographics: SeniorCitizen, Partner, Dependents
- Services: tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, TechSupport
- Account Info: Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- Target: Churn (Yes/No)
Saves raw dataset to 'raw_data/telco_customer_churn_raw.csv'.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
OUTPUT_RAW_CSV = os.path.join(RAW_DATA_DIR, "telco_customer_churn_raw.csv")

def generate_telecom_churn_data(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    
    customer_ids = [f"{np.random.randint(1000, 9999)}-{np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))}{np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))}" for _ in range(n_samples)]
    genders = np.random.choice(['Female', 'Male'], n_samples)
    senior_citizens = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    partners = np.random.choice(['Yes', 'No'], n_samples, p=[0.48, 0.52])
    dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70])
    
    # Tenure in months (1 to 72)
    tenure = np.random.randint(1, 73, n_samples)
    
    contracts = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.24, 0.21])
    internet_services = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.34, 0.44, 0.22])
    payment_methods = np.random.choice([
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ], n_samples)
    
    # Base monthly charges depending on internet service
    monthly_charges = []
    for service in internet_services:
        if service == 'Fiber optic':
            monthly_charges.append(np.round(np.random.uniform(70.0, 118.0), 2))
        elif service == 'DSL':
            monthly_charges.append(np.round(np.random.uniform(45.0, 75.0), 2))
        else:
            monthly_charges.append(np.round(np.random.uniform(18.0, 25.0), 2))
            
    monthly_charges = np.array(monthly_charges)
    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 10, n_samples), 2)
    total_charges = np.clip(total_charges, 18.0, None)
    
    # Introduce a few missing values in TotalCharges to simulate real-world data cleaning requirements
    total_charges_str = [str(tc) if np.random.rand() > 0.015 else " " for tc in total_charges]
    
    # Churn probability logic (Month-to-month + Fiber optic + high monthly charges + short tenure -> higher churn)
    churn_score = (
        (contracts == 'Month-to-month') * 1.5 +
        (internet_services == 'Fiber optic') * 1.0 +
        (tenure < 12) * 1.2 +
        (monthly_charges > 80) * 0.8 -
        (contracts == 'Two year') * 2.0
    )
    churn_prob = 1 / (1 + np.exp(-churn_score + 1.2))
    churn = np.where(np.random.rand(n_samples) < churn_prob, 'Yes', 'No')
    
    df = pd.DataFrame({
        'customerID': customer_ids,
        'gender': genders,
        'SeniorCitizen': senior_citizens,
        'Partner': partners,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10]),
        'InternetService': internet_services,
        'Contract': contracts,
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.60, 0.40]),
        'PaymentMethod': payment_methods,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges_str,
        'Churn': churn
    })
    
    # Duplicate 10 rows to simulate raw data duplicates
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    return df

def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    print("Generating raw Telco Customer Churn dataset...")
    df = generate_telecom_churn_data(n_samples=1200)
    df.to_csv(OUTPUT_RAW_CSV, index=False)
    print(f"Raw dataset successfully created with {len(df)} records at: {OUTPUT_RAW_CSV}")

if __name__ == "__main__":
    main()
