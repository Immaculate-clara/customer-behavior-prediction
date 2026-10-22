# Customer Behavior Prediction Platform

End-to-end machine-learning platform for a retail business: it predicts **which customers are about to leave (churn)**, **how much revenue they will bring in the next 90 days (customer lifetime value, CLV)**, and **which behavioural segment each customer belongs to**. Predictions are explained with SHAP and LIME, served through a FastAPI backend, stored in PostgreSQL and explored in a Streamlit dashboard. Everything runs in Docker.

Built for the Vantara Retail Solutions product requirements document (PRD) on the UCI **Online Retail II** dataset.

## Highlights

- Leakage-safe features: every feature is computed only from data before a cutoff date (2011-09-09); churn means "no purchase in the next 90 days".
- One shared, stratified 70 / 15 / 15 train / validation / test split for every model. Models and decision thresholds are chosen on validation data; the test set is used once.
- Classical models (logistic regression, decision tree, random forest, XGBoost, LightGBM, KNN), deep-learning models in PyTorch (ANN, LSTM) and an autoencoder for anomaly detection.
- K-Means customer segmentation, cross-checked with GMM, DBSCAN and hierarchical clustering.
- Explainability: SHAP, LIME and partial-dependence plots.
- REST API with validation, error handling and automatic documentation, plus a four-page dashboard.
- Automated tests with coverage reporting.

## Results

All numbers are on the held-out **test set** (5,262 customers overall; 15% held out for testing).

### Churn prediction (PRD targets: ROC-AUC >= 0.80, recall >= 0.70)

| Model | ROC-AUC | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| **XGBoost (used by the API)** | **0.813** | **0.875** | 0.755 | 0.811 | 0.768 |
| LightGBM | 0.815 | 0.879 | 0.739 | 0.803 | 0.756 |
| Random forest | 0.806 | 0.857 | 0.759 | 0.805 | 0.765 |
| Logistic regression | 0.802 | 0.723 | 0.788 | 0.754 | 0.733 |
| LSTM (PyTorch) | 0.804 | 0.828 | 0.741 | 0.782 | 0.738 |
| ANN (PyTorch) | 0.801 | 0.795 | 0.766 | 0.780 | 0.746 |
| Decision tree | 0.782 | 0.806 | 0.754 | 0.779 | 0.741 |
| KNN | 0.772 | 0.815 | 0.737 | 0.774 | 0.730 |

XGBoost was selected because it had the best **validation** ROC-AUC (0.814). Both PRD targets are met by XGBoost, LightGBM, random forest, logistic regression, and both neural networks.

### Customer lifetime value (PRD target: R-squared >= 0.60)

| Model | R-squared (log scale) | R-squared (revenue) | MAE | RMSE | Spearman |
|---|---|---|---|---|---|
| **XGBoost (used by the API)** | **0.356** | 0.384 | GBP 310 | GBP 882 | 0.575 |
| Random forest | 0.360 | 0.541 | GBP 299 | GBP 762 | 0.578 |
| LightGBM | 0.347 | 0.307 | GBP 318 | GBP 935 | 0.568 |
| Ridge | 0.341 | 0.206 | GBP 335 | GBP 1,001 | 0.572 |

**The R-squared target of 0.60 was not reached.** The best raw R-squared is 0.54 (random forest). Future spend is dominated by a few very large customers and by customers who simply do not return, which is hard to predict from purchase history alone. Models were ranked by log-scale R-squared on the validation set, which is more robust to those outliers. Ranking customers by expected value works reasonably (Spearman about 0.58), which is what retention prioritisation needs. See *Limitations* for ideas to improve it.

### Segmentation

Six customer segments were found with K-Means (k chosen by silhouette score between 4 and 6) and given business names such as *Champions*, *New & Promising*, *Seasonal Shoppers* and *Lost Customers*. GMM, DBSCAN and hierarchical clustering are reported as cross-checks. Customers that are unusual in their behaviour are flagged as outliers.

### Portfolio view (what the dashboard shows)

5,262 customers scored: average churn risk 52%, **1,782 high-risk customers (34%)**, expected revenue over the next 90 days of about GBP 994k, and GBP 850k of past spend sitting with high-risk customers.

## Architecture

```
data/raw (Excel) -> loader -> cleaner -> validator -> feature builder
                                                        |
        +-----------------------+-----------------------+
        v                       v                       v
  churn models              CLV models             segmentation
 (classical + deep)       (regression)             (K-Means ...)
        +-----------------------+-----------------------+
                                |
                     models_artifacts/ (joblib, .pt, csv, json)
                                |
                  FastAPI (api/) -- PostgreSQL (customer_scores)
                                |
                    Streamlit dashboard (frontend/)
```

## Project structure

```
.
├── api/                  FastAPI app: routers, schemas, services, portfolio
├── config/config.yaml    All paths, hyperparameters, thresholds and targets
├── data/                 raw / interim / processed (not committed)
├── docs/figures/         SHAP, LIME, PDP and clustering plots
├── frontend/             Streamlit dashboard (app.py + pages/)
├── models_artifacts/     Trained models, comparison tables, metadata
├── notebooks/            EDA, feature engineering and experiments
├── src/
│   ├── data/             loader, cleaner, validator, store_scores (PostgreSQL)
│   ├── features/         point-in-time feature engineering, category affinity
│   ├── models/           data_prep, evaluate, train_classical, train_clv, train_deep
│   ├── segmentation/     clustering
│   ├── explainability/   SHAP / LIME / PDP
│   └── utils/            config, logger, database
├── tests/                pytest suite
├── Dockerfile            API image
├── Dockerfile.frontend   Dashboard image
├── docker-compose.yml    PostgreSQL + API + dashboard
└── requirements.txt
```

## Getting started

### Requirements

- Python 3.11
- Docker Desktop (only for the PostgreSQL / container setup)
- The **Online Retail II** dataset from the UCI Machine Learning Repository, saved as `data/raw/online_retail_II.xlsx`

### Install

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Run the pipeline

Run each step once, in this order (each writes its output for the next one):

```powershell
python -m src.data.cleaner                    # clean raw transactions
python -m src.data.validator                  # check the cleaned data
python -m src.features.build_features         # customer feature table + labels
python -m src.models.train_classical          # churn models
python -m src.models.train_clv                # lifetime-value models
python -m src.models.train_deep               # ANN, LSTM, autoencoder
python -m src.segmentation.clustering         # customer segments
python -m src.explainability.explain          # SHAP, LIME, PDP plots
```

### Start the platform with Docker

Create a `.env` file (it is git-ignored):

```
POSTGRES_USER=cbp
POSTGRES_PASSWORD=cbp_password
POSTGRES_DB=customer_behavior
DATABASE_URL=postgresql+psycopg2://cbp:cbp_password@localhost:5432/customer_behavior
```

Then:

```powershell
docker compose up -d --build
python -m src.data.store_scores     # save all customer scores into PostgreSQL
```

| Service | Address |
|---|---|
| Dashboard | http://localhost:8501 |
| API documentation | http://localhost:8000/docs |
| PostgreSQL | localhost:5432 |

The API and dashboard containers read `data/` and `models_artifacts/` from your folders, so run the pipeline first.

### Run without Docker

```powershell
uvicorn api.main:app --port 8000
streamlit run frontend/app.py
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service status and loaded models |
| POST | `/predict/churn` | Churn probability, risk level and decision |
| POST | `/predict/clv` | Expected revenue for the next 90 days |
| POST | `/predict/segment` | Segment of a customer |
| POST | `/predict/batch` | All predictions for up to 1,000 customers |
| POST | `/explain/churn` | Top factors behind a churn prediction (SHAP) |
| GET | `/customers/{id}` | Full customer profile |
| GET | `/segments` | Segment summaries |
| GET | `/portfolio/summary` | KPIs and chart data for the dashboard |
| GET | `/portfolio/customers` | Customers ranked by retention priority |
| GET | `/models/comparison` | Model comparison tables |

A request identifies a customer either by `customer_id` or by a full set of feature values.

Example:

```powershell
curl -X POST http://localhost:8000/predict/churn -H "Content-Type: application/json" -d "{\"customer_id\": 12954}"
```

## Dashboard

- **Overview**: portfolio KPIs, risk distribution, churn risk and expected revenue by segment, and a "Who to contact first" table (filters and CSV download).
- **Customer Lookup**: one customer's churn gauge, expected revenue, segment, SHAP drivers and a recommended action.
- **Segments**: size, churn risk and revenue of every segment, with a behaviour heat map.
- **Model Performance**: churn and CLV model comparison against the PRD targets, plus explainability plots.

## Testing

```powershell
pytest
pytest --cov=src --cov=api --cov=frontend --cov-report=term-missing:skip-covered
```

The tests use small synthetic data and need neither the dataset nor trained models. The feature tests include explicit point-in-time leakage checks.

## Design decisions

- **No data leakage:** features only use transactions before the cutoff; the label looks 90 days ahead.
- **Threshold selection:** the churn decision threshold is chosen on validation data by balanced accuracy, subject to a minimum recall. Plain F1 is not used, because churners are the majority class and F1 would reward flagging almost everyone.
- **CLV on a log scale:** revenue is extremely skewed, so regressors are trained on `log1p(revenue)` and converted back.
- **Retention priority** = churn probability x past spend, so the team contacts customers who are both likely to leave and valuable.

## Limitations and future work

- The CLV R-squared target (0.60) is not met; a two-stage model (will the customer buy? then how much?) and more customer-level features are the most promising improvements.
- Customer country is not yet used as a model feature.
- The data covers a single UK online retailer over two years, so results may not transfer to other businesses.
- Models are trained once; there is no automated retraining or drift monitoring.

## Author

Immaculate Clara