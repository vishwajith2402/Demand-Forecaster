from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from demand_forecaster.data import generate_sample_data, generate_synthetic_supply_chain_data
from demand_forecaster.forecasting import (
    DEFAULT_FEATURES,
    ForecastConfig,
    build_forecast_plot,
    forecast_future,
    prepare_training_frame,
    train_model,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
SAMPLE_DATA = DATA_DIR / "sample_sales.csv"
SYNTHETIC_DATA = DATA_DIR / "synthetic_supply_chain_sales.csv"


st.set_page_config(
    page_title="AI Demand Forecaster",
    layout="wide",
)


def apply_ui_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --accent: #007f73;
            --accent-dark: #005f56;
            --ink: #172026;
            --muted: #5b6870;
            --line: #d8e2e0;
            --surface: #ffffff;
            --soft: #eef7f5;
            --warning: #b42318;
        }

        .stApp {
            background: #f7faf9;
            color: var(--ink);
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }

        .app-header {
            background: linear-gradient(135deg, #ffffff 0%, #e7f5f2 100%);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 28px 30px;
            margin-bottom: 22px;
        }

        .app-eyebrow {
            color: var(--accent-dark);
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .app-title {
            color: var(--ink);
            font-size: 2.15rem;
            font-weight: 760;
            line-height: 1.16;
            margin: 0 0 10px;
        }

        .app-subtitle {
            color: var(--muted);
            font-size: 1.02rem;
            max-width: 820px;
            margin: 0;
        }

        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
        }

        .insight-panel {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px 18px 12px;
            margin-bottom: 16px;
        }

        .status-good {
            color: var(--accent-dark);
            font-weight: 700;
        }

        .status-risk {
            color: var(--warning);
            font-weight: 700;
        }

        .small-note {
            color: var(--muted);
            font-size: 0.9rem;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_sample_data() -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    if not SAMPLE_DATA.exists():
        generate_sample_data(SAMPLE_DATA)
    return SAMPLE_DATA


def ensure_synthetic_data() -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    if not SYNTHETIC_DATA.exists():
        generate_synthetic_supply_chain_data(SYNTHETIC_DATA)
    return SYNTHETIC_DATA


@st.cache_data(show_spinner=False)
def load_csv(file_or_path) -> pd.DataFrame:
    return pd.read_csv(file_or_path)


def validate_input(df: pd.DataFrame) -> list[str]:
    required = {"date", "product_id", "sales", "promotion", "holiday"}
    missing = sorted(required.difference(df.columns))
    errors = []
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
    return errors


def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div class="app-eyebrow">Supply Chain AI</div>
            <div class="app-title">Demand Forecasting Dashboard</div>
            <p class="app-subtitle">
                Predict product demand, compare forecast accuracy, and estimate inventory requirements
                from historical sales, promotions, holidays, and pricing patterns.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(sample_path: Path, synthetic_path: Path):
    with st.sidebar:
        st.header("Data Source")
        uploaded = st.file_uploader("Upload sales CSV", type=["csv"])
        dataset_choice = st.radio(
            "Choose dataset",
            ["Synthetic supply chain", "Retail sample", "Upload CSV"],
            index=0,
        )

        if uploaded is None and dataset_choice == "Upload CSV":
            st.warning("Upload a CSV or choose a built-in dataset.")

        if dataset_choice == "Synthetic supply chain":
            st.caption(f"Loaded built-in file: {synthetic_path.name}")
        elif dataset_choice == "Retail sample":
            st.caption(f"Loaded built-in file: {sample_path.name}")
        elif uploaded is not None:
            st.caption(f"Loaded uploaded file: {uploaded.name}")
        else:
            st.caption("Using synthetic data until a CSV is uploaded.")

        st.divider()
        st.header("Forecast Settings")
        horizon = st.radio("Forecast horizon", [7, 14, 30], index=2, horizontal=True)
        test_days = st.slider("Backtest window", min_value=14, max_value=90, value=30, step=7)
        model_choice = st.selectbox("Model", ["Auto XGBoost / Gradient Boosting", "ARIMA baseline"])
        safety_stock_rate = st.slider("Safety stock", min_value=0, max_value=40, value=15, step=5)
        retrain = st.button("Train Forecast Model", type="primary", width="stretch")

        st.divider()
        st.markdown(
            """
            <div class="small-note">
            Required CSV columns: date, product_id, sales, promotion, holiday.
            Optional: price, store_id.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if dataset_choice == "Synthetic supply chain":
        source = synthetic_path
    elif dataset_choice == "Retail sample":
        source = sample_path
    elif uploaded is not None:
        source = uploaded
    else:
        source = synthetic_path
    return source, horizon, test_days, model_choice, safety_stock_rate, retrain


def render_dataset_controls(raw_df: pd.DataFrame):
    product_options = sorted(raw_df["product_id"].astype(str).unique())
    store_series = raw_df.get("store_id", pd.Series(["S1"]))
    store_options = ["All stores"] + sorted(store_series.astype(str).unique())

    selector_col_a, selector_col_b = st.columns([1, 1])
    with selector_col_a:
        product_id = st.selectbox("Product / SKU", product_options)
    with selector_col_b:
        store_id = st.selectbox("Store / Warehouse", store_options)

    filtered = raw_df[raw_df["product_id"].astype(str) == str(product_id)].copy()
    if store_id != "All stores" and "store_id" in filtered.columns:
        filtered = filtered[filtered["store_id"].astype(str) == store_id].copy()

    return product_id, store_id, filtered


def render_overview(raw_df: pd.DataFrame, filtered: pd.DataFrame) -> None:
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Selected Rows", f"{len(filtered):,}")
    col_b.metric("Products", raw_df["product_id"].nunique())
    col_c.metric("Stores", raw_df["store_id"].nunique() if "store_id" in raw_df.columns else "1")
    col_d.metric("Average Daily Sales", f"{filtered['sales'].mean():.1f}")

    st.markdown(
        f"""
        <div class="insight-panel">
            <strong>Date coverage:</strong> {filtered['date'].min().date()} to {filtered['date'].max().date()}
            <br>
            <span class="small-note">The model trains on historical rows for the selected product and store scope.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_inventory_recommendation(forecast: pd.DataFrame, horizon: int, safety_stock_rate: int) -> None:
    total_units = int(round(float(forecast["predicted_sales"].sum())))
    safety_stock = int(round(total_units * (safety_stock_rate / 100)))
    recommended_inventory = total_units + safety_stock
    avg_daily = total_units / max(horizon, 1)
    status = "Ready for planned demand" if safety_stock_rate >= 10 else "Low buffer"
    status_class = "status-good" if safety_stock_rate >= 10 else "status-risk"

    st.markdown(
        f"""
        <div class="insight-panel">
            <h3>Inventory Recommendation</h3>
            <p class="{status_class}">{status}</p>
            <p class="small-note">
            Keep enough inventory for expected demand plus a configurable safety buffer.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_a, metric_b = st.columns(2)
    metric_a.metric(f"Next {horizon} Days", f"{total_units:,} units")
    metric_b.metric("Average Per Day", f"{avg_daily:.1f} units")
    metric_c, metric_d = st.columns(2)
    metric_c.metric("Safety Stock", f"{safety_stock:,} units")
    metric_d.metric("Recommended Stock", f"{recommended_inventory:,} units")


def main() -> None:
    apply_ui_theme()
    render_header()

    sample_path = ensure_sample_data()
    synthetic_path = ensure_synthetic_data()
    source, horizon, test_days, model_choice, safety_stock_rate, retrain = render_sidebar(sample_path, synthetic_path)

    raw_df = load_csv(source)
    errors = validate_input(raw_df)
    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
    raw_df = raw_df.dropna(subset=["date"])
    product_id, store_id, filtered = render_dataset_controls(raw_df)

    if filtered.empty:
        st.warning("No rows are available for the selected product and store.")
        st.stop()

    use_arima = model_choice == "ARIMA baseline"
    config = ForecastConfig(horizon=int(horizon), test_days=int(test_days), use_arima=use_arima)
    run_key = (
        str(product_id),
        str(store_id),
        int(horizon),
        int(test_days),
        bool(use_arima),
        int(safety_stock_rate),
        len(filtered),
    )

    if "result" not in st.session_state or retrain or st.session_state.get("run_key") != run_key:
        with st.spinner("Training model and creating forecast..."):
            train_frame = prepare_training_frame(filtered)
            result = train_model(train_frame, config)
            forecast = forecast_future(train_frame, result, config)
            st.session_state.result = result
            st.session_state.forecast = forecast
            st.session_state.train_frame = train_frame
            st.session_state.run_key = run_key

    result = st.session_state.result
    forecast = st.session_state.forecast
    train_frame = st.session_state.train_frame

    overview_tab, forecast_tab, explain_tab, data_tab = st.tabs(
        ["Overview", "Forecast", "Explainability", "Data"]
    )

    with overview_tab:
        render_overview(raw_df, filtered)
        st.subheader("Model Accuracy")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", f"{result.metrics['mae']:.2f}")
        m2.metric("RMSE", f"{result.metrics['rmse']:.2f}")
        m3.metric("WMAPE", f"{result.metrics['wmape']:.2%}")
        m4.metric("Model", result.model_name)

        st.subheader("Backtest Preview")
        backtest_preview = result.backtest.copy()
        backtest_preview["date"] = backtest_preview["date"].dt.date
        backtest_preview["sales"] = backtest_preview["sales"].round(0).astype(int)
        backtest_preview["predicted_sales"] = backtest_preview["predicted_sales"].round(0).astype(int)
        st.dataframe(backtest_preview.tail(12), width="stretch", hide_index=True)

    with forecast_tab:
        left, right = st.columns([1.45, 0.85])
        with left:
            st.subheader("Demand Forecast")
            plot = build_forecast_plot(train_frame, forecast, result.backtest)
            st.pyplot(plot, clear_figure=True)

            display_forecast = forecast.copy()
            display_forecast["date"] = display_forecast["date"].dt.date
            display_forecast["predicted_sales"] = display_forecast["predicted_sales"].round(0).astype(int)
            st.dataframe(display_forecast, width="stretch", hide_index=True)

        with right:
            render_inventory_recommendation(forecast, int(horizon), int(safety_stock_rate))

            csv = forecast.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Forecast CSV",
                data=csv,
                file_name=f"forecast_product_{product_id}_{horizon}_days.csv",
                mime="text/csv",
                width="stretch",
            )

            MODEL_DIR.mkdir(exist_ok=True)
            model_path = MODEL_DIR / "latest_model.joblib"
            if st.button("Save trained model", width="stretch"):
                result.save(model_path)
                st.success(f"Model saved to {model_path}")

    with explain_tab:
        st.subheader("Prediction Drivers")
        if result.feature_importance is not None and not result.feature_importance.empty:
            importance = result.feature_importance.head(12)
            st.bar_chart(importance.set_index("feature")["importance"])
            st.caption("Higher values indicate stronger influence on model predictions.")
        else:
            st.info("Feature importance is not available for the selected baseline model.")

        st.subheader("How This Forecast Was Built")
        st.markdown(
            """
            The application cleans the sales history, creates calendar features, adds lagged sales,
            computes rolling averages, trains the selected model, tests it on the most recent holdout
            window, and then forecasts future demand recursively.
            """
        )

    with data_tab:
        st.subheader("Selected Training Data")
        preview_cols = ["date", "product_id", "sales"] + [c for c in DEFAULT_FEATURES if c in train_frame.columns]
        st.dataframe(train_frame[preview_cols].tail(60), width="stretch", hide_index=True)

        st.subheader("Raw Dataset Sample")
        raw_preview = raw_df.copy()
        raw_preview["date"] = raw_preview["date"].dt.date
        st.dataframe(raw_preview.head(100), width="stretch", hide_index=True)


def running_inside_streamlit() -> bool:
    return get_script_run_ctx(suppress_warning=True) is not None


def launch_with_streamlit() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
        check=False,
    )


if __name__ == "__main__" and not running_inside_streamlit():
    launch_with_streamlit()
elif __name__ == "__main__":
    main()
