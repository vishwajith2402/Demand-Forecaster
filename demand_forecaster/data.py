from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_data(output_path: str | Path, seed: int = 42) -> pd.DataFrame:
    """Create a realistic multi-product retail sales sample dataset."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    products = [101, 102, 103, 104, 105]
    stores = ["S1", "S2", "S3"]
    rows = []

    product_base = {101: 54, 102: 38, 103: 72, 104: 28, 105: 46}
    product_price = {101: 19.99, 102: 11.49, 103: 29.99, 104: 7.99, 105: 15.99}
    store_factor = {"S1": 1.0, "S2": 0.82, "S3": 1.18}

    for product_id in products:
        for store_id in stores:
            for date in dates:
                day_of_year = date.dayofyear
                weekly = 1.0 + 0.18 * np.sin(2 * np.pi * date.weekday() / 7)
                yearly = 1.0 + 0.22 * np.sin(2 * np.pi * day_of_year / 365)
                holiday = int((date.month, date.day) in {(1, 1), (8, 15), (10, 2), (12, 25)})
                promotion = int(rng.random() < (0.12 + 0.08 * (date.weekday() in [5, 6])))
                trend = 1 + ((date - dates[0]).days / len(dates)) * rng.uniform(0.03, 0.12)
                price_noise = rng.normal(0, 0.35)
                price = max(3.0, product_price[product_id] + price_noise - promotion * 1.2)
                demand = (
                    product_base[product_id]
                    * store_factor[store_id]
                    * weekly
                    * yearly
                    * trend
                    * (1.23 if promotion else 1.0)
                    * (1.35 if holiday else 1.0)
                )
                sales = max(0, int(round(rng.normal(demand, demand * 0.12))))
                rows.append(
                    {
                        "date": date,
                        "product_id": product_id,
                        "store_id": store_id,
                        "sales": sales,
                        "promotion": promotion,
                        "holiday": holiday,
                        "price": round(price, 2),
                    }
                )

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def generate_synthetic_supply_chain_data(output_path: str | Path, seed: int = 7) -> pd.DataFrame:
    """Create a richer synthetic dataset with supply-chain and market effects."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", "2025-03-31", freq="D")
    products = [
        {"product_id": "SKU-1001", "product_name": "Smart LED Bulb", "category": "Electronics", "base": 118, "price": 14.99},
        {"product_id": "SKU-1002", "product_name": "Protein Snack Box", "category": "Grocery", "base": 165, "price": 8.49},
        {"product_id": "SKU-1003", "product_name": "Cold Relief Syrup", "category": "Pharma", "base": 74, "price": 6.99},
        {"product_id": "SKU-1004", "product_name": "Cotton T-Shirt", "category": "Apparel", "base": 92, "price": 17.99},
        {"product_id": "SKU-1005", "product_name": "Kitchen Storage Set", "category": "Home", "base": 61, "price": 24.99},
        {"product_id": "SKU-1006", "product_name": "Bluetooth Speaker", "category": "Electronics", "base": 48, "price": 39.99},
        {"product_id": "SKU-1007", "product_name": "Organic Rice Pack", "category": "Grocery", "base": 142, "price": 12.99},
        {"product_id": "SKU-1008", "product_name": "Running Shoes", "category": "Apparel", "base": 36, "price": 59.99},
    ]
    warehouses = [
        {"store_id": "WH-NORTH", "warehouse_region": "North", "factor": 1.08},
        {"store_id": "WH-SOUTH", "warehouse_region": "South", "factor": 0.92},
        {"store_id": "WH-EAST", "warehouse_region": "East", "factor": 1.18},
        {"store_id": "WH-WEST", "warehouse_region": "West", "factor": 0.86},
    ]
    holiday_dates = {
        (1, 1),
        (1, 26),
        (8, 15),
        (10, 2),
        (11, 12),
        (12, 25),
        (12, 31),
    }
    category_seasonality = {
        "Electronics": {"festival": 1.32, "winter": 1.08, "summer": 0.95},
        "Grocery": {"festival": 1.16, "winter": 1.02, "summer": 1.04},
        "Pharma": {"festival": 1.02, "winter": 1.34, "summer": 0.90},
        "Apparel": {"festival": 1.24, "winter": 1.12, "summer": 1.08},
        "Home": {"festival": 1.18, "winter": 1.05, "summer": 0.96},
    }

    rows = []
    for product in products:
        product_growth = rng.uniform(0.05, 0.22)
        for warehouse in warehouses:
            promo_cooldown = 0
            for date in dates:
                day_index = (date - dates[0]).days
                month = date.month
                weekday = date.weekday()
                is_weekend = weekday in [5, 6]
                is_festival = month in [10, 11, 12]
                is_winter = month in [12, 1, 2]
                is_summer = month in [4, 5, 6]
                holiday = int((date.month, date.day) in holiday_dates)

                promo_probability = 0.08 + (0.07 if is_weekend else 0) + (0.05 if is_festival else 0)
                if promo_cooldown > 0:
                    promo_probability *= 0.35
                    promo_cooldown -= 1
                promotion = int(rng.random() < promo_probability)
                if promotion:
                    promo_cooldown = rng.integers(3, 9)

                market_trend = 1 + product_growth * (day_index / len(dates))
                weekly = 1 + 0.11 * np.sin(2 * np.pi * weekday / 7)
                annual = 1 + 0.16 * np.sin(2 * np.pi * date.dayofyear / 365)
                seasonal = category_seasonality[product["category"]]
                category_factor = 1.0
                if is_festival:
                    category_factor *= seasonal["festival"]
                if is_winter:
                    category_factor *= seasonal["winter"]
                if is_summer:
                    category_factor *= seasonal["summer"]

                competitor_discount = rng.uniform(0.0, 0.18)
                price_discount = rng.uniform(0.06, 0.22) if promotion else rng.uniform(-0.02, 0.04)
                price = max(2.5, product["price"] * (1 - price_discount))
                competitor_price = max(2.5, product["price"] * (1 - competitor_discount))
                price_pressure = 1 + np.clip((competitor_price - price) / product["price"], -0.18, 0.18)

                weather_index = float(np.clip(rng.normal(0.55, 0.18), 0, 1))
                if product["category"] == "Pharma" and weather_index < 0.35:
                    category_factor *= 1.16
                if product["category"] == "Grocery" and weather_index > 0.72:
                    category_factor *= 1.08

                stockout = int(rng.random() < 0.018)
                stockout_factor = 0.55 if stockout else 1.0

                demand = (
                    product["base"]
                    * warehouse["factor"]
                    * market_trend
                    * weekly
                    * annual
                    * category_factor
                    * price_pressure
                    * (1.28 if promotion else 1.0)
                    * (1.22 if holiday else 1.0)
                    * stockout_factor
                )
                sales = max(0, int(round(rng.normal(demand, max(5, demand * 0.13)))))

                rows.append(
                    {
                        "date": date,
                        "product_id": product["product_id"],
                        "product_name": product["product_name"],
                        "category": product["category"],
                        "store_id": warehouse["store_id"],
                        "warehouse_region": warehouse["warehouse_region"],
                        "sales": sales,
                        "promotion": promotion,
                        "holiday": holiday,
                        "price": round(price, 2),
                        "competitor_price": round(competitor_price, 2),
                        "weather_index": round(weather_index, 3),
                        "stockout": stockout,
                    }
                )

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
