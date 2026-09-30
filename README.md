# 🔮 Project 2: Customer Churn Prediction — Machine Learning & Pipeline

**Customer Churn Prediction** is an end-to-end Machine Learning classification project designed to identify at-risk customers in a telecom service environment before they switch providers. The project implements data loading, rigorous cleaning, feature engineering, Logistic Regression & Random Forest model training, priority metrics evaluation (FPR, TPR, FNR, ROC-AUC), and an interactive CLI prediction script.

---

## 📁 Repository Structure

```
customer_churn/
├── raw_data/
│   └── telco_customer_churn_raw.csv      # Raw customer records dataset
├── cleaned_data/
│   └── telco_customer_churn_cleaned.csv  # Standardized CSV without duplicates/nulls
├── output/
│   ├── 1_confusion_matrix.png            # Classification heatmap
│   ├── 2_roc_auc_curve.png               # ROC Curve with AUC score
│   ├── 3_feature_importance.png          # Key predictors ranking
│   └── 4_churn_distribution.png          # Target class ratio plot
├── load_data.py                          # Dataset generation & loader script
├── clean_data.py                         # Data cleaning & standardization script
├── train_model.py                        # Model training & metrics evaluation script
├── predict_interactive.py                # Interactive CLI prediction script
├── customer_churn_notebook.ipynb         # Complete interactive Jupyter Notebook
├── churn_model.pkl                       # Serialized trained model & scaler pipeline
└── README.md                             # Project documentation
```

---

## ⚙️ Project Pipeline & Methodology

### 1. Data Cleaning (`clean_data.py`)
- **Duplicates**: Identified and removed duplicate rows.
- **Missing Values**: Handled missing/empty string entries in `TotalCharges` by converting to numeric floats and imputing missing records with the median.
- **Unwanted Columns**: Removed non-informative identifier `customerID`.
- **Categorical Standardization**: Standardized `Contract` types (`month-to-month`, `one-year`, `two-year`), trimmed whitespace, and fixed casing issues.

### 2. Feature Engineering & Preprocessing (`train_model.py`)
- Created derived features: `avg_monthly_per_tenure` ($\text{TotalCharges} / (\text{tenure} + 1)$) and `is_high_charge`.
- Encoded categorical variables using `LabelEncoder`.
- Scaled features using `StandardScaler` to prepare for distance-based and linear models.

### 3. Model Building & Evaluation Priority Metrics
- Trained **Logistic Regression** and **Random Forest Classifiers**.
- Evaluated key performance indicators specified in course guidelines:
  - **Recall (True Positive Rate - TPR)**: Measures ability to catch churners.
  - **False Positive Rate (FPR)**: Rate of misclassifying loyal customers.
  - **False Negative Rate (FNR)**: Crucial retention metric $\text{FN} / (\text{TP} + \text{FN})$ measuring missed churners.
  - **ROC-AUC Score**: Overall classification discrimination quality.
  - **Confusion Matrix**: Visual breakdown of TP, FP, TN, FN.

### 4. Interactive Prediction Script (`predict_interactive.py`)
Provides an interactive command-line interface that prompts for customer parameters (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`, `Contract`, `InternetService`, etc.) and outputs:
- **Prediction**: Customer likely to **CHURN** or **STAY**.
- **Probability**: Confidence percentage (e.g. `84.50% Churn Probability`).
- **Risk Assessment**: `HIGH RISK` or `LOW RISK` with actionable business advice!

---

## 🚀 How to Run the Project

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Step 1: Run the Pipeline
```bash
py load_data.py
py clean_data.py
py train_model.py
```

### Step 2: Test Interactive Predictions
```bash
py predict_interactive.py
```

---

## 🌐 GitHub Upload Instructions

```bash
git init
git add .
git commit -m "Complete Project 2: Customer Churn Prediction Machine Learning Pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/customer-churn-prediction.git
git push -u origin main
```
