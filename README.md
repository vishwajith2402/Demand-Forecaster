# AI-Based Supply Chain Demand Forecasting System

This project is an executable Streamlit application for SKU-level supply chain demand forecasting. It predicts future product demand from historical sales, promotions, holidays, price, and time-based patterns.

## Features

- Streamlit dashboard with sidebar controls and tabbed pages
- Built-in synthetic supply-chain dataset with SKUs, warehouses, promotions, holidays, price movement, competitor pressure, weather index, and stockout events
- Forecast next 7, 14, or 30 days of demand
- Train an XGBoost model with automatic fallback to scikit-learn Gradient Boosting
- Optional ARIMA baseline model
- Multi-product and store-level filtering
- Lag features, rolling averages, trend indicators, weekday, month, and holiday signals
- Accuracy metrics: MAE, RMSE, and WMAPE
- Forecast visualization with historical sales and backtest predictions
- Inventory requirement estimate with safety stock
- Feature importance for explainable predictions
- CSV upload and forecast export
- Model saving with `joblib`
- PyInstaller build script for `.exe` packaging

## Dataset Format

Required columns:

| Column | Description |
| --- | --- |
| `date` | Sales date |
| `product_id` | Product or SKU identifier |
| `sales` | Quantity sold |
| `promotion` | 1 if product was on offer, otherwise 0 |
| `holiday` | 1 for special day or holiday, otherwise 0 |

Optional columns:

| Column | Description |
| --- | --- |
| `price` | Product price |
| `store_id` | Store or warehouse identifier |

Example:

```csv
date,product_id,store_id,sales,promotion,holiday,price
2023-01-01,101,S1,45,1,0,19.99
2023-01-02,101,S1,50,0,0,19.99
```

If no CSV is uploaded, the app can use either built-in dataset:

- `data/synthetic_supply_chain_sales.csv`: richer synthetic supply-chain data created for this project
- `data/sample_sales.csv`: smaller retail sample data

## Run Locally

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the application:

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal. The dashboard includes:

- Overview tab for dataset coverage and accuracy metrics
- Forecast tab for demand graph, prediction table, inventory estimate, model saving, and CSV export
- Explainability tab for feature importance and model workflow
- Data tab for selected training rows and raw dataset preview

## Build Windows Executable

After installing dependencies, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```

The executable output will be created in the `dist` directory. When opened, it launches the Streamlit demand forecasting application.

## Project Structure

```text
Demand_Forecaster/
  app.py
  build_exe.ps1
  requirements.txt
  README.md
  demand_forecaster/
    __init__.py
    data.py
    forecasting.py
  data/
    sample_sales.csv
    synthetic_supply_chain_sales.csv
  models/
```

## Forecasting Workflow

1. Load uploaded CSV or generated sample data.
2. Clean dates and aggregate daily sales.
3. Create calendar, lag, rolling average, and trend features.
4. Train XGBoost or Gradient Boosting on historical demand.
5. Backtest on the latest holdout window.
6. Generate recursive future forecasts.
7. Visualize predictions and estimate inventory needs.
