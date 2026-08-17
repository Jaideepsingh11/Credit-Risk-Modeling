# Credit Risk Assessment

A machine learning based credit risk assessment application that evaluates loan applicants and assigns them to one of four risk segments — **P1 (lowest risk) to P4 (highest risk)**.

The project combines bureau-level customer information, feature engineering, statistical analysis, and an XGBoost classification model into a Flask web application that can be used to score individual applicants.

---

## Overview

Credit risk assessment involves identifying applicants who are more likely to default while maintaining a consistent and data-driven evaluation process.

This project uses two bureau case-study datasets containing customer-level trade-line, delinquency, enquiry, and demographic information. The data is cleaned and transformed into model-ready features, followed by feature selection and model training using XGBoost.

The final model achieves approximately **78% accuracy on the test set**, with performance varying across the four risk segments. P3 is the most challenging segment to classify, which reflects the underlying class distribution and characteristics of the dataset.

The trained model is integrated into a Flask application where users can enter applicant information and receive an instant risk assessment.

---

## Key Features

* Credit risk classification into **P1, P2, P3, and P4**
* XGBoost-based machine learning model
* Data cleaning and preprocessing pipeline
* Feature selection and scaling
* Saved model artifacts for direct prediction
* Flask-based web interface
* JSON API for programmatic predictions
* Reproducible model training pipeline
* Production-ready WSGI entry point using Gunicorn

---

## Project Structure

```text
credit-risk-app/
│
├── data/
│   ├── case_study1.xlsx
│   └── case_study2.xlsx
│
├── model/
│   ├── train_model.py
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── feature_info.json
│
├── app/
│   ├── app.py
│   ├── wsgi.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
│
├── requirements.txt
├── Procfile
├── .gitignore
└── README.md
```

### Folder responsibilities

**`data/`**
Contains the original case-study datasets used during model development.

**`model/`**
Contains the training pipeline and the saved artifacts required for prediction.

**`app/`**
Contains the Flask application, HTML templates, CSS, and production WSGI configuration.

The web application loads the trained model and preprocessing artifacts directly rather than retraining the model for every prediction.

---

## Machine Learning Workflow

The overall workflow is:

```text
Raw Bureau Data
       │
       ▼
Data Cleaning
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Feature Engineering
       │
       ▼
Feature Selection
       │
       ▼
Train / Test Split
       │
       ▼
Feature Scaling
       │
       ▼
XGBoost Classifier
       │
       ▼
Risk Segment Prediction
       │
       ▼
Flask Web Application
```

The training pipeline uses a fixed random seed and stratified splitting to make the experiments reproducible.

The trained objects are saved using Joblib so that the application can load them directly without repeating the training process.

---

## Model

The project uses **XGBoost** for multi-class classification.

The target variable represents four credit-risk segments:

| Segment | Interpretation        |
| ------- | --------------------- |
| P1      | Lowest risk           |
| P2      | Low to moderate risk  |
| P3      | Moderate to high risk |
| P4      | Highest risk          |

The model was evaluated using a held-out test set and achieved approximately **78% test accuracy**.

Performance is not uniform across all four classes. P3 is comparatively harder to identify, which is useful from a business perspective because it highlights where additional data or improved feature engineering could improve the model.

---

## Web Application

The Flask application provides a simple interface for entering applicant information and generating a risk assessment.

The application:

1. Accepts applicant and bureau information.
2. Applies the same preprocessing used during model training.
3. Loads the trained XGBoost model.
4. Generates the predicted risk segment.
5. Displays the result through the web interface.

The application also exposes a JSON endpoint for programmatic predictions.

### API Endpoint

```text
POST /api/predict
```

Example request:

```json
{
  "NETMONTHLYINCOME": 45000,
  "AGE": 34,
  "MARITALSTATUS": "Married"
}
```

Additional model features can be supplied in the request body.

Fields that are not provided are assigned appropriate default values by the application.

---

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/credit-risk-modeling.git
cd credit-risk-modeling
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

If you want to reproduce the model artifacts from the source datasets:

```bash
cd model
python train_model.py
cd ..
```

This generates:

```text
model.pkl
scaler.pkl
label_encoder.pkl
feature_info.json
```

### 5. Start the Flask application

```bash
python app/app.py
```

Open:

```text
http://127.0.0.1:5000
```

The application is now ready to score applicants.

---

## Tech Stack

**Programming**

* Python

**Machine Learning**

* XGBoost
* Scikit-learn
* Statsmodels
* NumPy
* Pandas
* SciPy

**Data**

* Excel
* OpenPyXL

**Web Application**

* Flask
* HTML
* CSS

**Deployment**

* Gunicorn
* Procfile
* Compatible with cloud platforms such as Render and Railway

---

## Model Artifacts

The application uses four saved artifacts:

| File                | Purpose                                                             |
| ------------------- | ------------------------------------------------------------------- |
| `model.pkl`         | Trained XGBoost model                                               |
| `scaler.pkl`        | Feature scaling transformation                                      |
| `label_encoder.pkl` | Converts risk labels between numerical and original representations |
| `feature_info.json` | Stores information about the features expected by the model         |

Keeping these artifacts separately from the web application makes it possible to retrain the model independently and deploy a new model without rebuilding the entire application.

---

## Deployment

The project can be deployed using platforms that support Python and Flask applications.

The included `Procfile` provides a production entry point using Gunicorn.

For example:

```text
gunicorn --chdir app wsgi:app --bind 0.0.0.0:$PORT
```

The trained model artifacts should be available in the repository when deploying the application.

For a portfolio project, the simplest workflow is:

```text
GitHub
   ↓
Cloud Deployment
   ↓
Flask Application
   ↓
Credit Risk Prediction
```

---

## Future Improvements

There are several directions in which the project could be extended:

* Hyperparameter optimization for XGBoost
* Probability-based risk scoring
* Model explainability using SHAP
* ROC-AUC and class-wise evaluation
* Improved handling of class imbalance
* Automated model retraining
* Model monitoring and drift detection
* Applicant-level explanation of the predicted risk
* Interactive dashboards for portfolio-level credit analysis
* Deployment with Docker and cloud infrastructure

---

## Disclaimer

This project is developed as a machine learning and portfolio application using case-study credit bureau data.

The predictions are intended for demonstration and analytical purposes and should not be used as the sole basis for real-world lending or credit decisions.

---

## Author

**Jaideep Singh**

Chemical Engineering | Machine Learning | Data Analytics

This project was developed to explore the application of machine learning to credit risk assessment, from raw bureau data and feature engineering through model development and deployment.
