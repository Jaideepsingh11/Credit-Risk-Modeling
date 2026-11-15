# Credit Risk Assessment & Loan Default Prediction

Analysis of 32,581 loans to find what drives borrower default, and to predict defaults using **Logistic Regression** and **Random Forest**.

## Files
| File | What it is |
|---|---|
| `Credit_Risk_Analysis.ipynb` | All code: cleaning, EDA, statistics, feature engineering, models, evaluation |
| `Project_Summary.pdf` | 3-page summary of findings |
| `Model_Comparison.csv` | Cross-validation and test metrics for both models |
| `visualizations/` | 12 simple charts produced by the notebook |
| `credit_risk_dataset.csv` | Dataset |

## Approach
1. **Data cleaning:** removed 165 duplicates and 7 impossible values (age up to 144, 123 years of employment). Missing values are filled inside the model pipeline to avoid leakage.
2. **EDA:** distributions, default rate by each feature, correlation and combined-effect heatmaps.
3. **Statistics:** Mann-Whitney test with effect size (numeric), chi-square with Cramér's V (categorical).
4. **Features:** log income, grade as 1–7, prior default 1/0, high loan-to-income flag (≥30%), missing-employment flag, one-hot encoding, scaling.
5. **Models:** 80/20 stratified split, 5-fold cross-validation, `class_weight="balanced"` for the 22% default imbalance.

## Key findings
- **22% of loans default.**
- **Loan grade:** default rate rises from 10% (A) to 59% (D) and 98% (G).
- **Loan-to-income ratio:** about 12–22% default below 30%, about 70% above.
- **Income and housing:** lowest-income group defaults at 43% vs 9% for the highest; renters 32% vs mortgage holders 13%.
- **Prior default on file** doubles risk (38% vs 18%).
- **Age and credit history length** have almost no effect.

## Results (test set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.811 | 0.547 | **0.797** | 0.649 | 0.882 |
| Random Forest | **0.932** | **0.912** | 0.760 | **0.829** | **0.938** |

Random Forest wins: far fewer false alarms (104 vs 935). Logistic Regression catches slightly more defaults but wrongly flags many good borrowers.

**Top factors (Random Forest):** income, loan-to-income ratio, loan grade, interest rate.

## Limitations
Grade and interest rate come from the lender's own risk assessment, so findings are associations, not causes. No time dimension in the data.

## Run it
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn jupyter
jupyter notebook Credit_Risk_Analysis.ipynb
```
