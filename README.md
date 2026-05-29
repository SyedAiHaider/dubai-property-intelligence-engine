# Dubai Property Intelligence Engine: Machine Learning & Explainable AI (XAI) for Residential Property Valuation

The Dubai Property Intelligence Engine is an end-to-end data science framework designed to automate and interpret residential property valuations in Dubai. Utilizing over 289,000 verified transaction records from the Dubai Land Department (DLD), the framework replaces traditional, linear hedonic pricing with high-performance ensemble learning models (**Random Forest & XGBoost**). 

To bridge the gap between machine learning performance and institutional transparency, this repository integrates **SHAP (Shapley Additive Explanations)** for Explainable AI (XAI)—dismantling the "black-box" nature of advanced algorithms by calculating the exact monetary contribution (in AED) of every property feature.

---

## 🚀 Key Features

* **AI-Driven Valuation Engine:** Provides instant, real-time market appraisals for apartments and villas across premium Dubai sectors (e.g., Business Bay, Downtown Dubai, Palm Jumeirah).
* **Explainable AI (XAI) Dashboard:** Incorporates global and local SHAP interpretation plots to show stakeholders exactly how size (`procedure_area`) and location intelligence (`area_mean_price`) weigh on valuations.
* **Feature Engineering Pipeline:** Includes automated calculation transformations such as target-encoded neighborhood premiums and Unit Density metrics (sqft/room).
* **Interactive Financial Forecaster:** Connects live predictive inferences with integrated financial tools to evaluate real-world down payments, interest rates, and estimated monthly outflows (EMI).

---

## 📊 Model Performance Summary

The predictive pipeline evaluates four distinct algorithmic approaches. Tree-based ensemble methods significantly outperformed traditional linear baselines, explaining more than **80% of market variance**:

| Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | R-Squared ($R^2$) |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 622,978.48 | 993,309.35 | 0.5910 |
| **Decision Tree** | 416,702.61 | 769,547.10 | 0.7545 |
| **Random Forest** | 360,266.04 | 674,874.90 | **0.8112** |
| **XGBoost** | 378,000.25 | 683,187.84 | **0.8065** |

---

## 🛠️ Technology Stack

* **Core Language:** Python 3.12+
* **Data Processing & Analytics:** Pandas, NumPy, Scikit-Learn
* **Gradient Boosting:** XGBoost
* **Model Explainability:** SHAP (Game-Theoretic Shapley Values)
* **Web Deployment Framework:** Streamlit
* **Serialization:** Joblib

---

## 💻 Installation and Local Deployment

To run the Streamlit PropTech web application locally on your machine, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/SyedAiHaider/dubai-property-intelligence-engine.git](https://github.com/SyedAiHaider/dubai-property-intelligence-engine.git)
cd dubai-property-intelligence-engine
