# Telecom Customer Churn Prediction

> Supervised machine learning project to predict customer churn for a telecom operator.  
> Completed as part of the **MAL course – ENSIIE** (December 2025).  
> 📄 A detailed report (in French) is available in the `report/` folder.

---

## Objective

Customer churn — when a client switches to a competitor — is a major challenge for telecom operators. Retaining an existing customer is significantly cheaper than acquiring a new one. This project aims to **predict the probability of churn** from demographic, financial, and behavioural features, and to identify the key drivers of customer attrition.

---

## Key Results

| Model | F1 (test) | AUC (test) |
|---|---|---|
| Gradient Boosting ✅ | **0.58** | **0.86** |
| Decision Tree | 0.55 | 0.82 |
| Random Forest | — | — |
| Logistic Regression | — | — |
| KNN | — | — |

**Top features:** `Age` > `NumOfProducts` > `IsActiveMember` > `Geography_Germany`

**Fairness analysis (Gender):** both models generate more churn alerts for female customers (GB: ~18.5%) than for male customers (~9.8%), with moderate TPR/FPR gaps — no major bias detected.

---

## Project Structure

```
churn-prediction/
├── README.md
├── requirements.txt
├── data/
│   └── celldata.csv           # Dataset (8,000 customers, 11 features)
├── notebooks/
│   └── churn_prediction.ipynb
└── report/
    └── rapport_churn.pdf      # Detailed report (French)
```

---

##  Methodology

1. **Exploratory Data Analysis (EDA)**
   - Distribution of continuous and categorical variables by churn status
   - Correlation heatmap
   - Identification of the most discriminating features

2. **Preprocessing**
   - One-hot encoding (`Geography`)
   - Binary encoding (`Gender`)
   - Standardisation of continuous features for LogReg and KNN
   - Stratified train/test split (70/30)

3. **Modelling**
   - Comparison of 5 classifiers: LogReg, KNN, Decision Tree, Random Forest, Gradient Boosting
   - Testing with and without anomaly detection (IsolationForest)
   - Selection criterion: **F1-score** (imbalanced dataset)

4. **Hyperparameter Tuning**
   - `GridSearchCV` with 5-fold cross-validation
   - Decision Tree: `max_depth=9` → CV F1 = 0.57
   - Gradient Boosting: `n_estimators=200`, `learning_rate=0.05` → CV F1 = 0.58

5. **Fairness Analysis**
   - Sensitive attribute: `Gender`
   - Criteria: **Independence** (demographic parity), **Separation** (TPR/FPR by gender), **Sufficiency** (conditional calibration)

---

##  Business Recommendations

- Prioritise **older customers** and **German customers** for retention campaigns
- Proactively target customers with **3 products** (churn rate ~83%)
- Re-engage **inactive members** with personalised offers
- Use the Decision Tree's explicit rules to communicate risk factors to marketing teams

---

##  Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green)

---

##  Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/churn-prediction.git
cd churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook notebooks/churn_prediction.ipynb
```

> ⚠️ Place `celldata.csv` in the `data/` folder before running the notebook.

---

##  Author

**Halimata Ndiaye** – Engineering student at ENSIIE  
[LinkedIn](https://www.linkedin.com/in/halimata-ndiaye-514763268/) · [GitHub](https://github.com/Halimatand)
