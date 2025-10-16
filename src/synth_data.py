from __future__ import annotations
import numpy as np, pandas as pd
from datetime import datetime, timedelta

def generate_apple_sales(n_rows: int = 1000, base_demand: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime.now() - timedelta(days=i) for i in range(n_rows)][::-1]
    months = np.array([d.month for d in dates])
    weekend = np.array([1 if d.weekday()>=5 else 0 for d in dates], dtype=int)
    holiday = rng.choice([0,1], n_rows, p=[0.97,0.03]).astype(int)
    temp = rng.uniform(10, 35, n_rows)
    rain = rng.exponential(5, n_rows)
    price = rng.uniform(0.5, 3.0, n_rows)
    harvest = np.sin(2*np.pi*(months-3)/12) + np.sin(2*np.pi*(months-9)/12)
    price = price - 0.5*harvest
    inflation = 1 + (pd.Series(dates).dt.year - pd.Timestamp(dates[0]).year) * 0.03
    demand = (base_demand - 50*price + 50*harvest + 300*weekend + rng.normal(0,50,n_rows)) * inflation.to_numpy()
    df = pd.DataFrame({
        "date": dates,
        "average_temperature": temp,
        "rainfall": rain,
        "weekend": weekend,
        "holiday": holiday,
        "price_per_kg": price,
        "previous_days_demand": pd.Series(demand).shift(1).bfill(),
        "demand": demand
    })
    return df
