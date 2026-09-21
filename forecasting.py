"""
forecasting.py
--------------
Demand forecasting logic.

Baseline model (PRIMARY, used everywhere in the app including the reorder
calculation): trailing N-day moving average of historical Units_Sold,
projected forward as a flat daily rate across the selected forecast period.
This is the simple, explainable baseline requested in the brief.

Optional comparison model: scikit-learn LinearRegression fitted on a
trailing window of Units_Sold vs. day-index, used ONLY to show a trend-based
alternative for discussion purposes. Per the assessment brief, this is not
claimed to be superior to the baseline - no back-testing/accuracy comparison
has been performed in this Sprint 2 prototype, so the two are shown side by
side for transparency, not as a performance claim.

Data-leakage note: Simulated_True_Demand is NEVER used as a model input.
It exists only in the source dataset for later evaluation purposes.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

MOVING_AVERAGE_WINDOW = 7  # 7-day moving average, as requested in the brief


def _series_for(df: pd.DataFrame, store: str, product_id: str) -> pd.DataFrame:
    mask = (df["Store"] == store) & (df["Product_ID"] == product_id)
    return df.loc[mask, ["Date", "Units_Sold"]].sort_values("Date").reset_index(drop=True)


def baseline_moving_average_forecast(
    df: pd.DataFrame,
    store: str,
    product_id: str,
    as_of_date: pd.Timestamp,
    forecast_days: int,
    window: int = MOVING_AVERAGE_WINDOW,
) -> dict:
    """
    Baseline forecast: average of the last `window` days of Units_Sold
    (ending at as_of_date), held flat across the forecast horizon.

    Returns a dict with the historical series, forecast series and the
    average daily rate used (this rate also feeds the reorder calculation).
    """
    series = _series_for(df, store, product_id)
    series = series[series["Date"] <= as_of_date]

    if series.empty:
        return {
            "avg_daily_rate": 0.0,
            "historical": series,
            "forecast_dates": [],
            "forecast_values": [],
        }

    trailing = series.tail(window)
    avg_daily_rate = float(trailing["Units_Sold"].mean())

    forecast_dates = pd.date_range(
        start=as_of_date + pd.Timedelta(days=1), periods=forecast_days, freq="D"
    )
    forecast_values = [avg_daily_rate] * forecast_days

    return {
        "avg_daily_rate": avg_daily_rate,
        "window_used": window,
        "historical": series,
        "forecast_dates": list(forecast_dates),
        "forecast_values": forecast_values,
    }


def linear_trend_forecast(
    df: pd.DataFrame,
    store: str,
    product_id: str,
    as_of_date: pd.Timestamp,
    forecast_days: int,
    trailing_history_days: int = 60,
) -> dict:
    """
    OPTIONAL comparison model: scikit-learn LinearRegression fitted on the
    trailing `trailing_history_days` of Units_Sold, used to project a trend.

    Model choice rationale (for the Sprint 2 write-up): Linear Regression is
    a simple, transparent, low-data-requirement model well suited to a short
    university-project timeline, and it is easy to explain to a non-technical
    audience (a straight trend line) - appropriate for demonstrating
    scikit-learn integration without over-engineering the Sprint 2 design.
    It is NOT claimed to outperform the baseline; no accuracy testing has
    been conducted in this sprint.
    """
    series = _series_for(df, store, product_id)
    series = series[series["Date"] <= as_of_date].tail(trailing_history_days)

    if len(series) < 5:
        return {"avg_daily_rate": 0.0, "forecast_dates": [], "forecast_values": []}

    x = np.arange(len(series)).reshape(-1, 1)
    y = series["Units_Sold"].values

    model = LinearRegression()
    model.fit(x, y)

    future_x = np.arange(len(series), len(series) + forecast_days).reshape(-1, 1)
    preds = model.predict(future_x)
    preds = np.clip(preds, a_min=0, a_max=None)  # demand cannot be negative

    forecast_dates = pd.date_range(
        start=as_of_date + pd.Timedelta(days=1), periods=forecast_days, freq="D"
    )

    return {
        "avg_daily_rate": float(np.mean(preds)) if len(preds) else 0.0,
        "forecast_dates": list(forecast_dates),
        "forecast_values": list(preds),
        "trend_slope_per_day": float(model.coef_[0]),
    }
