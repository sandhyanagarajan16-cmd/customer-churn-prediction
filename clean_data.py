"""
Data Cleaning Script for Project 2: Customer Churn Prediction
Implements all workflow steps specified in course material:
- Drop duplicate records
- Remove unwanted identifier columns (customerID)
- Clean missing values & empty spaces in 'TotalCharges', 'tenure', 'MonthlyCharges'
- Standardize categorical variables (contract types, fix spacing and casing)
Saves cleaned dataset to 'cleaned_data/telco_customer_churn_cleaned.csv'.
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(__file__)
RAW_CSV = os.path.join(BASE_DIR, "raw_data", "telco_customer_churn_raw.csv")
CLEANED_CSV = os.path.join(BASE_DIR, "cleaned_data", "telco_customer_churn_cleaned.csv")

def clean_churn_dataset(input_csv, output_csv):
    print(f"Loading raw dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    initial_len = len(df)
    
    # 1. Drop duplicates
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_len - len(df)} duplicate records.")
    
    # 2. Drop unwanted columns (customerID)
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
        print("Removed unwanted column: 'customerID'")
        
    # 3. Handle missing values & whitespace issues in numeric columns
    # TotalCharges contains strings/spaces that need coercion to float
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    
    # Impute missing TotalCharges with median
    missing_total = df['TotalCharges'].isna().sum()
    if missing_total > 0:
        median_val = df['TotalCharges'].median()
        df['TotalCharges'] = df['TotalCharges'].fillna(median_val)
        print(f"Imputed {missing_total} missing values in TotalCharges with median value ({median_val:.2f}).")
        
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(df['tenure'].median()).astype(int)
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(df['MonthlyCharges'].median())
    
    # 4. Standardize categorical variables (casing, trimming whitespace)
    cat_columns = df.select_dtypes(include=['object']).columns
    for col in cat_columns:
        df[col] = df[col].astype(str).str.strip()
        
    # Standardize Contract values: 'month-to-month', 'one-year', 'two-year'
    df['Contract'] = df['Contract'].str.lower().replace({
        'month-to-month': 'month-to-month',
        'one year': 'one-year',
        'two year': 'two-year'
    })
    
    # Standardize InternetService values
    df['InternetService'] = df['InternetService'].str.lower()
    
    # Standardize Churn target variable: 'Yes'/'No' to binary 1/0
    df['Churn_Binary'] = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    
    print(f"\nCleaned dataset shape: {df.shape}")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Cleaned dataset saved successfully to: {output_csv}")
    return df

def main():
    if not os.path.exists(RAW_CSV):
        print(f"Error: Raw CSV not found at {RAW_CSV}. Run load_data.py first.")
        return
    clean_churn_dataset(RAW_CSV, CLEANED_CSV)

if __name__ == "__main__":
    main()
