"""
Model Building & Evaluation Script for Project 2: Customer Churn Prediction
Implements all requirements specified in course slides:
- Feature engineering & scaling (StandardScaler, Label Encoding)
- Logistic Regression & Decision Tree / Random Forest models
- Priority Evaluation Metrics:
    - False Positive Rate (FPR) & True Positive Rate (TPR)
    - False Negative Rate (FNR)
    - ROC-AUC Curve
    - Confusion Matrix
    - Precision, Recall, F1-score, Support
Saves trained pipeline to 'churn_model.pkl' and plots to 'output/'.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score
)

BASE_DIR = os.path.dirname(__file__)
CLEANED_CSV = os.path.join(BASE_DIR, "cleaned_data", "telco_customer_churn_cleaned.csv")
MODEL_PKL = os.path.join(BASE_DIR, "churn_model.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

sns.set_theme(style="whitegrid")

def train_and_evaluate_churn_model(input_csv):
    print(f"Loading cleaned dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # 1. Feature Engineering
    print("\n--- Feature Engineering ---")
    df['avg_monthly_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['is_high_charge'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)
    
    # Select feature columns
    cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'InternetService', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_per_tenure', 'is_high_charge']
    
    # Encode categorical features
    df_encoded = df.copy()
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
        
    X = df_encoded[num_cols + cat_cols]
    y = df_encoded['Churn_Binary']
    
    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Model Building (Logistic Regression & Random Forest)
    print("\n--- Training Models ---")
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Use Best Performing Model (Random Forest handles non-linear patterns best)
    y_pred = rf_model.predict(X_test_scaled)
    y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    # 3. Evaluation & Metrics Priority Calculation (Screenshot 6)
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0  # False Negative Rate (Crucial for Churn!)
    auc_score = roc_auc_score(y_test, y_prob)
    
    print("\n==========================================")
    print("      MODEL EVALUATION METRICS SUMMARY    ")
    print("==========================================")
    print(f"Accuracy:            {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision:           {precision_score(y_test, y_pred):.4f}")
    print(f"Recall (TPR):        {TPR:.4f}")
    print(f"False Positive Rate: {FPR:.4f}")
    print(f"False Negative Rate: {FNR:.4f}  <-- (Key Retention Metric!)")
    print(f"F1-Score:            {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC Score:       {auc_score:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stay (0)', 'Churn (1)']))
    
    # 4. Save Model Pipeline
    model_data = {
        'model': rf_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_cols': num_cols + cat_cols,
        'num_cols': num_cols,
        'cat_cols': cat_cols
    }
    with open(MODEL_PKL, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Trained model pipeline saved to: {MODEL_PKL}")
    
    # 5. Generate & Save Charts to output/
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Chart 1: Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
    plt.title('Confusion Matrix', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Chart 2: ROC-AUC Curve
    fpr_vals, tpr_vals, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr_vals, tpr_vals, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR / Recall)')
    plt.title('ROC-AUC Curve', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_roc_auc_curve.png'), dpi=300)
    plt.close()
    
    # Chart 3: Feature Importance
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance', ascending=False)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
    plt.xlabel('Importance Weight')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_feature_importance.png'), dpi=300)
    plt.close()
    
    # Chart 4: Churn Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn', palette='Set2')
    plt.title('Customer Churn Class Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Churn Status')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '4_churn_distribution.png'), dpi=300)
    plt.close()
    
    print(f"\nAll evaluation plots saved to output directory: {OUTPUT_DIR}")

def main():
    if not os.path.exists(CLEANED_CSV):
        print(f"Error: Cleaned CSV not found at {CLEANED_CSV}. Run clean_data.py first.")
        return
    train_and_evaluate_churn_model(CLEANED_CSV)

if __name__ == "__main__":
    main()
