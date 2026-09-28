# Telco Customer Churn Prediction and Retention Project

A supervised machine learning project that predicts which telecommunications customers are likely to cancel their subscription, and translates those predictions into a prioritised retention action list.

**Dataset:** IBM Watson Analytics Telco Customer Churn with 7,043 customers, 21 raw features, 26.6% churn rate.

---

## Project Structure

```
telco-churn-ml-lab/
├── app.py                          # Streamlit churn risk dashboard
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
│       ├── telco_clean.csv         # Cleaned, type-cast dataset
│       ├── telco_featured.csv      # 38-feature engineered dataset
│       └── churn_scored_customers.csv  # Every customer scored with risk tier
├── models/
│   ├── best_churn_model.joblib             # Best baseline model
│   ├── best_churn_model_optimised.joblib   # Best tuned model (production)
│   ├── optimised_model_features.json       # 44-feature list for inference
│   ├── optimised_model_metrics.json        # Tuned model test metrics
│   └── supervised_sklearn_metrics.json     # Baseline model test metrics
└── notebooks/
    ├── 00_setup_and_eda.ipynb          # Data loading, cleaning, exploration
    ├── 01a_feature_engineering.ipynb   # 38-feature build: encodings, ratios, flags
    ├── 01b_data preprocessing.ipynb    # Train/test split, SMOTE, scaling
    ├── 02_supervised_sklearn.ipynb     # Baseline LR, RF, GB with default params
    ├── 03_optimising_models.ipynb      # +6 interaction features, RandomizedSearchCV, XGBoost
    └── 04_business_output.ipynb        # Churn scoring, risk tiers, CSV export
```

---

## Key Findings from EDA

- **Tenure is the strongest single predictor.** Customers who churn have a median tenure of 10 months vs 38 months for retained customers. The first year is the highest-risk window.
- **Contract type dominates churn rate.** Month-to-month customers churn at 42%, one-year at 11%, two-year at 3%. Contract lock-in is the most effective structural retention mechanism.
- **Monthly charges correlate with churn, but the relationship is non-linear.** High-spend customers churn more, but only in the absence of protective add-ons, which suggests perceived value matters more than absolute spend.
- **Protective services (online security, tech support, device protection) are strongly protective.** Customers with zero add-on services churn at 52% when they have internet access, vs 24% for customers with at least one add-on.

---

## Feature Engineering

The raw 21-column dataset was expanded to **44 features** across two stages.

**Stage 1 (notebook 01a), 38 features:** ordinal contract encoding, log-transformed charges, tenure groupings, add-on service counts, binary flags for auto-payment and electronic check, interaction terms for senior customers living alone, and early-tenure risk flags.

**Stage 2 (notebook 03), 6 additional interaction features:**

| Feature | Signal |
|---|---|
| `fiber_no_protection` | Fiber optic + no protective add-ons: high bill, low stickiness |
| `fiber_month_to_month` | Fiber + month-to-month: intersection of two top individual predictors |
| `contract_payment_risk` | Month-to-month + electronic check: no contractual or payment commitment |
| `internet_no_addons` | Has internet but zero add-ons: fixes a non-monotonic signal in the 38-feature set |
| `charge_per_service` | MonthlyCharges ÷ (services + 1): perceived value proxy |
| `paperless_no_auto` | Paperless billing + manual payment: sees every bill, low switching friction |

All models were trained on SMOTE-balanced training data (50/50 class split) with `StandardScaler` applied to continuous features, fitted on the training portion only.

---

## Models Trained

Hyperparameter tuning used `RandomizedSearchCV` with 5-fold `StratifiedKFold`, scored on **F1 (churn class)** to reflect the business priority of catching churners while keeping precision actionable.

| Model | ROC-AUC | Recall (churn) | F1 (churn) |
|---|---|---|---|
| LR Tuned | **0.8294** | 0.6711 | **0.6070** |
| LR Baseline | 0.8273 | 0.6845 | 0.6045 |
| GB Baseline | 0.8227 | **0.7246** | 0.6002 |
| RF Baseline | 0.8187 | 0.6658 | 0.6058 |
| RF Tuned | 0.8129 | 0.6337 | 0.5859 |
| GB Tuned | 0.8049 | 0.6364 | 0.5833 |
| XGBoost Tuned | 0.7988 | 0.6150 | 0.5750 |
| XGBoost Default | 0.7905 | 0.5909 | 0.5567 |

Test set: 1,407 customers, 26.6% churn rate (real-world distribution, no SMOTE applied at inference).

---

## Model Selection

**Production model: Gradient Boosting (Baseline)** - selected on recall (0.7246), the highest across all eight models.

The business context here is marketing targeting: a false negative (missed churner) means lost revenue with no opportunity to intervene, while a false positive (wrongly flagged retained customer) costs only a retention call. When the cost of misses exceeds the cost of false alarms, recall is the right optimisation target.

LR Tuned is the better choice if ROC-AUC (0.8294) or F1 (0.6070) are preferred, for example, if the retention budget is constrained and precision matters more.

---

## Business Output

Notebook 04 scores all 7,032 customers and assigns a **risk tier** based on predicted churn probability:

| Tier | Probability | Recommended Action |
|---|---|---|
| **Critical** | ≥ 70% | Immediate outreach: personalised retention offer within 48 hours |
| **High** | 50–70% | Proactive contact within 1 week: contract upgrade or discount incentive |
| **Medium** | 30–50% | Quarterly check-in: soft upsell to add-ons or longer contract |
| **Low** | < 30% | No action: standard communications cadence |

A decision threshold of **0.3** (below the default 0.5) is used for the binary `predicted_churn` flag to maximise recall in the flagged group. The tier column gives the retention team finer priority within that group.

The scored output is saved to `data/processed/churn_scored_customers.csv` with columns: `customer_id`, `churn_probability`, `risk_tier`, `predicted_churn`, `tenure`, `MonthlyCharges`, `Contract`.

---

## Streamlit Dashboard

`app.py` provides an interactive view of the scored customer file.

**Run it:**

```powershell
# activating the virtual environment first
.venv\Scripts\Activate.ps1

streamlit run app.py
```

Opens at `http://localhost:8501`. Features:

- Four KPI cards: total customers, predicted churners, critical risk count, average churn probability
- Bar chart of customer count by risk tier
- Churn probability distribution histogram with threshold markers at 0.3, 0.5, 0.7
- Grouped bar chart of average tenure and monthly charges by tier
- Sidebar filters: risk tier, contract type, probability range
- Sortable customer detail table with progress-bar probability column

> Run notebook `04_business_output.ipynb` first to generate `churn_scored_customers.csv`.

---

## How to Run

1. **Clone the repository** and open the folder in VS Code.

2. **Create a virtual environment and install dependencies:**

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python -m ipykernel install --user --name=telco-ml --display-name "Python (telco-ml)"
   ```

3. **Download the dataset** and save it as `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
   Source: [Kaggle: IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

4. **Run notebooks in order**, selecting the `Python (telco-ml)` kernel in each:

   ```
   00_setup_and_eda.ipynb
   01a_feature_engineering.ipynb
   01b_data preprocessing.ipynb
   02_supervised_sklearn.ipynb
   03_optimising_models.ipynb
   04_business_output.ipynb
   ```

5. **Launch the dashboard:**

   ```powershell
   streamlit run app.py
   ```

---

## Key Learnings

- **Feature engineering outperformed hyperparameter tuning.** Adding the 6 interaction features in notebook 03 produced a larger lift than any tuning run. Diagnosing *why* a feature was misleading the model (e.g. the non-monotonic `num_add_on_services` signal) led directly to the right fix.
- **All models converged to a similar performance ceiling (~0.82–0.83 ROC-AUC), suggesting the bottleneck is data richness rather than model choice.** Adding behavioural data (usage patterns, support contact frequency, plan changes) would likely break through that ceiling more effectively than further tuning.
- **Metric selection must be driven by the business cost of errors, not statistical convention.** Accuracy is misleading on imbalanced data. ROC-AUC measures ranking quality but says nothing about the decision threshold. The choice between recall and F1 is a business decision about the relative cost of false negatives vs false positives, and that answer changes by use case.
