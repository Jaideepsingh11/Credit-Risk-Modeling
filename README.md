# Credit Risk Prediction & Explainable Loan Default Assessment

An end-to-end Machine Learning project for predicting the probability of loan default using applicant financial and credit information.

The project covers the complete ML lifecycle:

- Data cleaning and validation
- Exploratory Data Analysis
- Feature preprocessing
- Class imbalance handling
- Logistic Regression baseline
- XGBoost classification
- Stratified Cross-Validation
- XGBoost hyperparameter tuning
- Model evaluation
- Probability calibration
- SHAP-based explainability
- Model serialization
- FastAPI deployment
- Real-time credit risk prediction

The final system takes a loan applicant's information as input and returns a predicted probability of default along with a **High Risk / Low Risk** classification.

---

# 1. Project Objective

The objective of this project is to build a machine learning system capable of identifying applicants who are more likely to default on a loan.

Instead of relying only on a binary prediction, the system produces a probability of default, allowing the prediction to be interpreted as a measure of credit risk.

The deployed system exposes the trained model through a FastAPI backend so that predictions can be generated for new loan applications in real time.

---

# 2. Dataset

The dataset contains information about individual loan applications.

The original dataset contains:

- **32,581 loan applications**
- **12 features**

After removing duplicate records, **32,416 records** remained.

The dataset contains numerical and categorical variables.

## Features

| Feature | Description | Type |
|---|---|---|
| `person_age` | Age of the applicant | Numerical |
| `person_income` | Annual income of the applicant | Numerical |
| `person_home_ownership` | Home ownership status | Categorical |
| `person_emp_length` | Employment length | Numerical |
| `loan_intent` | Purpose of the loan | Categorical |
| `loan_grade` | Loan grade | Categorical |
| `loan_amnt` | Loan amount | Numerical |
| `loan_int_rate` | Loan interest rate | Numerical |
| `loan_status` | Loan default status | Target |
| `loan_percent_income` | Loan amount as percentage of income | Numerical |
| `cb_person_default_on_file` | Previous default history | Categorical |
| `cb_person_cred_hist_length` | Length of credit history | Numerical |

### Target Variable

`loan_status`

```text
0 → No Default
1 → Default
3. Data Cleaning

Several validation and cleaning steps were performed before model development.

Duplicate Removal

Duplicate records were removed.

Original records:          32,581
After duplicate removal:  32,416
Applicant Age

Applicant ages were restricted to a realistic range:

18 ≤ age ≤ 100
Employment Length

Employment length was checked against applicant age and extreme values were removed.

person_emp_length ≤ person_age
person_emp_length ≤ 60
Loan Amount

Loan amounts were required to be positive.

loan_amnt > 0

The notebook also checks the observed interest-rate range.

4. Machine Learning Pipeline

The project evaluates both a traditional linear baseline and a tree-based ensemble model.

The two main models are:

Logistic Regression
XGBoost

The XGBoost model is the primary model used for the final system.

5. Preprocessing

Different preprocessing strategies were used depending on the model.

Numerical Features

For XGBoost, missing numerical values are handled using median imputation.

No feature scaling is required for XGBoost.

Categorical Features

Categorical variables are processed using:

Missing Value Imputation
        ↓
One-Hot Encoding

OneHotEncoder(handle_unknown="ignore") is used so that unseen categories do not break the prediction pipeline.

The preprocessing and model are combined into a single Scikit-Learn pipeline.

6. Handling Class Imbalance

Credit default prediction is an imbalanced classification problem because the number of non-default applications is significantly larger than the number of default applications.

The XGBoost model therefore uses class weighting through:

scale_pos_weight

This increases the importance of the minority/default class during model training.

This is particularly important because simply maximizing accuracy can result in a model that performs poorly at identifying actual defaulters.

7. Baseline Model — Logistic Regression

Logistic Regression was used as a baseline model.

A 5-fold Stratified Cross-Validation strategy was used.

Logistic Regression Cross-Validation Results
Metric	Score
ROC-AUC	0.871
Accuracy	0.812
Precision	0.545
Recall	0.778
F1 Score	0.641

The baseline provides a reference point for evaluating the more complex XGBoost model.

8. XGBoost Model

XGBoost was selected as the main machine learning algorithm because it can effectively model nonlinear relationships and interactions between applicant characteristics.

The initial XGBoost pipeline consists of:

Raw Applicant Data
        ↓
Numerical Imputation
        ↓
Categorical Imputation
        ↓
One-Hot Encoding
        ↓
XGBoost Classifier
        ↓
Default Probability
        ↓
Risk Classification

The initial model uses:

max_depth = 5
learning_rate = 0.1
n_estimators ≈ 300
scale_pos_weight = calculated class weight
9. Cross-Validation

To obtain a more reliable estimate of model performance, Stratified 5-Fold Cross-Validation was used.

The same class distribution is approximately maintained across the folds.

The evaluation metrics were:

ROC-AUC
Accuracy
Precision
Recall
F1 Score
10. XGBoost Cross-Validation Results

The initial XGBoost model achieved the following mean 5-fold cross-validation performance:

Metric	XGBoost
ROC-AUC	0.9394
Accuracy	0.9087
Precision	0.7913
Recall	0.7857
F1 Score	0.7880

These results show a substantial improvement over the Logistic Regression baseline.

Comparison
Metric	Logistic Regression	XGBoost
ROC-AUC	0.871	0.939
Accuracy	0.812	0.909
Precision	0.545	0.791
Recall	0.778	0.786
F1	0.641	0.788
11. Hyperparameter Tuning

The XGBoost model was further optimized using:

RandomizedSearchCV

The search used:

5-fold Stratified Cross-Validation
150 parameter combinations
750 total model fits

The optimization metric was:

Average Precision

This metric was selected because the problem involves an imbalanced classification target.

12. Hyperparameters Tuned

The following XGBoost parameters were optimized:

n_estimators
max_depth
learning_rate
subsample
colsample_bytree
min_child_weight
gamma
Search Space
n_estimators:
150 → 600

max_depth:
3 → 8

learning_rate:
0.01 → 0.51

subsample:
0.60 → 1.00

colsample_bytree:
0.60 → 1.00

min_child_weight:
1 → 9

gamma:
0 → 5
13. Best XGBoost Hyperparameters

The best configuration obtained from the randomized search was:

Parameter	Value
n_estimators	366
max_depth	5
learning_rate	0.13167
subsample	0.84530
colsample_bytree	0.93743
min_child_weight	7
gamma	2.39136

The best cross-validation Average Precision score was:

0.90
14. Final XGBoost Test Performance

The tuned XGBoost model was evaluated on the held-out test set.

Test Set
Total test samples: 6,305

Class distribution:

Class 0: 4,943
Class 1: 1,362
Results
Metric	XGBoost
Accuracy	0.92
Precision	0.81
Recall	0.81
F1 Score	0.81
Classification Report
Class	Precision	Recall	F1
No Default (0)	0.95	0.95	0.95
Default (1)	0.81	0.81	0.81
Overall	0.92	0.92	0.92

The tuned model therefore provides substantially stronger minority-class performance than the Logistic Regression baseline.

15. Precision-Recall Analysis

Because default prediction is an imbalanced classification problem, Precision-Recall analysis was also performed.

The project evaluates:

Precision
Recall
F1 Score

across different probability thresholds.

The purpose is to investigate whether changing the default classification threshold can provide a better balance between:

Identifying actual defaulters
Avoiding false positives
16. Probability Calibration

Raw machine learning probabilities are not always perfectly calibrated.

For a credit-risk system, probability calibration is important because the output probability should ideally correspond to the observed frequency of default.

For example:

Predicted probability ≈ 0.70

should ideally represent a group of applicants where approximately 70% actually default.

The project therefore applies:

CalibratedClassifierCV

using:

method = "sigmoid"
cv = 5

This performs sigmoid/Platt-style probability calibration.

The process is:

Tuned XGBoost
      ↓
Sigmoid Calibration
      ↓
Calibrated Default Probability

The calibration curves compare:

Uncalibrated XGBoost
vs
Calibrated XGBoost
17. Model Explainability with SHAP

The project uses SHAP (SHapley Additive exPlanations) to understand the predictions generated by XGBoost.

The trained XGBoost classifier is extracted from the preprocessing pipeline and passed to:

shap.TreeExplainer

SHAP values are calculated for the test set.

This allows the model to answer questions such as:

Which features are driving risk?
Which features increase predicted default probability?
Which features decrease predicted default probability?
Why did a particular applicant receive a specific prediction?
18. SHAP Analysis

Two types of explanations are generated.

Global Explanation

A SHAP summary plot is used to understand the overall importance and impact of features across the test set.

Individual Explanation

A SHAP waterfall plot is generated for an individual applicant.

The waterfall plot shows how each feature contributes to moving the prediction away from the model's baseline expectation.

This makes the model more interpretable than treating XGBoost as a completely black-box classifier.

19. Model Serialization

After training, the calibrated XGBoost model is saved using Joblib.

credit_risk_model.pkl

The classification threshold is also saved:

best_threshold.pkl

These files allow the trained model to be loaded directly by the FastAPI application without retraining the model every time the server starts.

20. FastAPI Deployment

The trained model is deployed using FastAPI.

The backend:

Loads the trained model
Loads the classification threshold
Receives applicant information
Validates the input using Pydantic
Converts the input into a Pandas DataFrame
Generates a default probability
Applies the classification threshold
Returns the risk classification

The model and threshold are loaded during the FastAPI application lifecycle.

FastAPI Startup
      ↓
Load credit_risk_model.pkl
      ↓
Load best_threshold.pkl
      ↓
Server Ready
21. FastAPI Input Schema

The API accepts the following applicant attributes:

person_age
person_income
person_home_ownership
person_emp_length
loan_intent
loan_grade
loan_amnt
loan_int_rate
loan_percent_income
cb_person_default_on_file
cb_person_cred_hist_length

The Pydantic model validates the incoming request before it reaches the prediction logic.

22. Prediction Endpoint

The main prediction endpoint is:

POST /predict

The API receives the applicant information and returns a prediction.

Internally, the endpoint:

Request
   ↓
Pydantic Validation
   ↓
Pandas DataFrame
   ↓
Trained ML Pipeline
   ↓
Default Probability
   ↓
Threshold Comparison
   ↓
Risk Classification

The implementation uses the trained model's:

predict_proba()

method to obtain the probability of class 1.

The API then compares that probability against the stored threshold.

23. API Request

Example request:

{
    "person_age": 28,
    "person_income": 60000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 10000,
    "loan_int_rate": 12.5,
    "loan_percent_income": 0.17,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 6
}
24. API Response

The API returns:

{
    "default_probability": 0.73,
    "default_prediction": 1,
    "threshold": 0.50,
    "Result": "High Risk"
}
Response Fields
Field	Description
default_probability	Predicted probability of default
default_prediction	Binary prediction
threshold	Classification threshold used
Result	Human-readable risk category

The classification is:

prediction = 1
→ High Risk

prediction = 0
→ Low Risk

The FastAPI implementation loads both the trained model and threshold at application startup and returns these prediction fields from /predict.

25. FastAPI Documentation

Once the server is running, FastAPI automatically provides interactive API documentation.

Swagger UI:

http://127.0.0.1:8000/docs

The API can be tested directly through the Swagger interface.

26. Project Architecture
                    CREDIT RISK SYSTEM
                           │
                           ▼
                   Loan Application Data
                           │
                           ▼
                    Data Cleaning
                           │
                           ▼
                    Preprocessing
                  ┌────────┴────────┐
                  │                 │
             Numerical          Categorical
             Imputation         Imputation
                  │                 │
                  │            One-Hot Encoding
                  └────────┬────────┘
                           │
                           ▼
                     XGBoost Model
                           │
                           ▼
                  Hyperparameter Tuning
                           │
                           ▼
                    Best XGBoost Model
                           │
                           ▼
                  Probability Calibration
                           │
                           ▼
                  Calibrated Probability
                           │
                           ▼
                     Risk Prediction
                           │
                           ▼
                    FastAPI Endpoint
                           │
                           ▼
              High Risk / Low Risk Result
27. Project Structure
credit_risk_app/
│
├── Credit_Risk.ipynb
│       └── Complete ML development,
│           evaluation and explainability
│
├── main.py
│       └── FastAPI backend
│
├── credit_risk_model.pkl
│       └── Saved calibrated XGBoost model
│
├── best_threshold.pkl
│       └── Saved classification threshold
│
├── requirements.txt
│       └── Python dependencies
│
├── static/
│       └── Frontend/static application files
│
└── README.md
