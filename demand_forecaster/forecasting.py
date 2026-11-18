from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency fallback
    XGBRegressor = None


try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:  # pragma: no cover - optional dependency fallback
    ARIMA = None


DEFAULT_FEATURES = [
    "promotion",
    "holiday",
    "price",
    "day",
    "month",
    "year",
    "weekday",
    "is_weekend",
    "quarter",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_7",
    "rolling_14",
    "rolling_30",
    "sales_trend_7",
]


@dataclass
class ForecastConfig:
    horizon: int = 30
    test_days: int = 30
    use_arima: bool = False


@dataclass
class ForecastResult:
    model: Any
    model_name: str
    metrics: dict[str, float]
    features: list[str]
    backtest: pd.DataFrame
    feature_importance: pd.DataFrame | None

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])
    prepared["promotion"] = prepared.get("promotion", 0).fillna(0).astype(int)
    prepared["holiday"] = prepared.get("holiday", 0).fillna(0).astype(int)
    if "price" not in prepared.columns:
        prepared["price"] = prepared["sales"].mean()
    prepared["price"] = prepared["price"].ffill().bfill().fillna(0)

    group_cols = ["date"]
    aggregations = {
        "sales": "sum",
        "promotion": "max",
        "holiday": "max",
        "price": "mean",
    }
    if "product_id" in prepared.columns:
        aggregations["product_id"] = "first"
    if "store_id" in prepared.columns:
        aggregations["store_id"] = "first"

    daily = prepared.groupby(group_cols, as_index=False).agg(aggregations).sort_values("date")
    full_dates = pd.DataFrame({"date": pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")})
    daily = full_dates.merge(daily, on="date", how="left")
    daily["sales"] = daily["sales"].fillna(0)
    daily["promotion"] = daily["promotion"].fillna(0).astype(int)
    daily["holiday"] = daily["holiday"].fillna(0).astype(int)
    daily["price"] = daily["price"].ffill().bfill().fillna(0)

    daily["day"] = daily["date"].dt.day
    daily["month"] = daily["date"].dt.month
    daily["year"] = daily["date"].dt.year
    daily["weekday"] = daily["date"].dt.weekday
    daily["is_weekend"] = daily["weekday"].isin([5, 6]).astype(int)
    daily["quarter"] = daily["date"].dt.quarter

    for lag in [1, 7, 14, 30]:
        daily[f"lag_{lag}"] = daily["sales"].shift(lag)
    for window in [7, 14, 30]:
        daily[f"rolling_{window}"] = daily["sales"].shift(1).rolling(window).mean()

    daily["sales_trend_7"] = daily["rolling_7"] - daily["rolling_14"]
    numeric_cols = daily.select_dtypes(include=[np.number]).columns
    daily[numeric_cols] = daily[numeric_cols].fillna(0)
    return daily


def train_model(df: pd.DataFrame, config: ForecastConfig) -> ForecastResult:
    if len(df) < max(45, config.test_days + 10):
        raise ValueError("At least 45 daily rows are recommended for a reliable forecast.")

    if config.use_arima:
        return _train_arima(df, config)

    features = [feature for feature in DEFAULT_FEATURES if feature in df.columns]
    split_date = df["date"].max() - pd.Timedelta(days=config.test_days)
    train = df[df["date"] <= split_date].copy()
    test = df[df["date"] > split_date].copy()

    if len(test) == 0:
        test = df.tail(config.test_days).copy()
        train = df.iloc[: -len(test)].copy()

    model = _make_regressor()
    model.fit(train[features], train["sales"])
    predictions = np.clip(model.predict(test[features]), 0, None)
    metrics = _metrics(test["sales"], predictions)

    backtest = test[["date", "sales"]].copy()
    backtest["predicted_sales"] = predictions

    importance = _feature_importance(model, features)
    return ForecastResult(
        model=model,
        model_name=model.__class__.__name__,
        metrics=metrics,
        features=features,
        backtest=backtest,
        feature_importance=importance,
    )


def forecast_future(df: pd.DataFrame, result: ForecastResult, config: ForecastConfig) -> pd.DataFrame:
    if result.model_name == "ARIMA":
        future_values = np.clip(result.model.forecast(steps=config.horizon), 0, None)
        future_dates = pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=config.horizon, freq="D")
        return pd.DataFrame({"date": future_dates, "predicted_sales": np.asarray(future_values)})

    history = df.copy().sort_values("date").reset_index(drop=True)
    future_rows = []
    last_price = float(history["price"].tail(14).mean()) if "price" in history.columns else 0.0
    promotion_rate = float(history["promotion"].tail(60).mean()) if "promotion" in history.columns else 0.0

    for step in range(config.horizon):
        next_date = history["date"].max() + pd.Timedelta(days=1)
        row = {
            "date": next_date,
            "promotion": int(step % 7 in [5, 6] and promotion_rate > 0.1),
            "holiday": int((next_date.month, next_date.day) in {(1, 1), (8, 15), (10, 2), (12, 25)}),
            "price": last_price,
            "day": next_date.day,
            "month": next_date.month,
            "year": next_date.year,
            "weekday": next_date.weekday(),
            "is_weekend": int(next_date.weekday() in [5, 6]),
            "quarter": next_date.quarter,
        }
        sales_series = history["sales"]
        for lag in [1, 7, 14, 30]:
            row[f"lag_{lag}"] = float(sales_series.iloc[-lag]) if len(sales_series) >= lag else float(sales_series.mean())
        for window in [7, 14, 30]:
            row[f"rolling_{window}"] = float(sales_series.tail(window).mean())
        row["sales_trend_7"] = row["rolling_7"] - row["rolling_14"]

        feature_frame = pd.DataFrame([row])
        prediction = float(np.clip(result.model.predict(feature_frame[result.features])[0], 0, None))
        row["sales"] = prediction
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        future_rows.append({"date": next_date, "predicted_sales": prediction})

    return pd.DataFrame(future_rows)


def build_forecast_plot(history: pd.DataFrame, forecast: pd.DataFrame, backtest: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    recent = history.tail(180)
    ax.plot(recent["date"], recent["sales"], label="Historical sales", color="#1f77b4", linewidth=2)
    if not backtest.empty:
        ax.plot(backtest["date"], backtest["predicted_sales"], label="Backtest prediction", color="#ff7f0e", linewidth=2)
    ax.plot(forecast["date"], forecast["predicted_sales"], label="Future forecast", color="#2ca02c", linewidth=2.5)
    ax.fill_between(forecast["date"], forecast["predicted_sales"] * 0.9, forecast["predicted_sales"] * 1.1, color="#2ca02c", alpha=0.12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand units")
    ax.set_title("Demand Forecast")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _make_regressor():
    if XGBRegressor is not None:
        return XGBRegressor(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.045,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
    return GradientBoostingRegressor(random_state=42, n_estimators=250, learning_rate=0.045, max_depth=3)


def _train_arima(df: pd.DataFrame, config: ForecastConfig) -> ForecastResult:
    if ARIMA is None:
        raise ImportError("statsmodels is required for the ARIMA baseline.")
    split_date = df["date"].max() - pd.Timedelta(days=config.test_days)
    train = df[df["date"] <= split_date].copy()
    test = df[df["date"] > split_date].copy()
    model = ARIMA(train["sales"].astype(float), order=(7, 1, 1)).fit()
    predictions = np.clip(model.forecast(steps=len(test)), 0, None)
    metrics = _metrics(test["sales"], predictions)
    backtest = test[["date", "sales"]].copy()
    backtest["predicted_sales"] = np.asarray(predictions)
    full_model = ARIMA(df["sales"].astype(float), order=(7, 1, 1)).fit()
    return ForecastResult(
        model=full_model,
        model_name="ARIMA",
        metrics=metrics,
        features=[],
        backtest=backtest,
        feature_importance=None,
    )


def _metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    denominator = max(float(np.abs(actual).sum()), 1.0)
    wmape = float(np.abs(actual - predicted).sum() / denominator)
    return {"mae": float(mae), "rmse": float(rmse), "wmape": wmape}


def _feature_importance(model: Any, features: list[str]) -> pd.DataFrame | None:
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        return None
    frame = pd.DataFrame({"feature": features, "importance": importance})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)
