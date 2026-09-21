"""
ml_forecast.py
---------------
A real, backtested machine learning demand forecast, built to sit alongside
(not silently replace) the 7-day moving-average baseline used everywhere
else in the app.

Model: scikit-learn RandomForestRegressor, trained per SKU on network-wide
daily totals (summed across all 6 stores), using lag/rolling/calendar
features engineered from historical Units_Sold. Random Forest was chosen
because it handles non-linear demand patterns (weekday effects, promotion
spikes) without needing manual feature scaling, trains fast on a dataset
this size, and is reasonably explainable via feature importances - all
practical fits for a Sprint 2 prototype.

Honesty / data-leakage rules (same as forecasting.py):
- Simulated_True_Demand is NEVER used as a feature or target.
- The train/test split is CHRONOLOGICAL (last N days held out as test),
  never random, so the model is never trained on data from its own future.
- Accuracy is reported for BOTH the baseline and the ML model on the exact
  same held-out test window, so any claim of one beating the other is based
  on a real, like-for-like backtest rather than assumption.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

TEST_WINDOW_DAYS = 60  # chronological hold-out for backtesting
N_ESTIMATORS = 200
RANDOM_STATE = 42


def _network_daily_series(df: pd.DataFrame, sku: str) -> pd.DataFrame:
    """Network-wide (all 6 stores summed) daily Units_Sold and Promotion rate for one SKU."""
    sub = df[df["Product_ID"] == sku].copy()
    daily = sub.groupby("Date").agg(
        Units_Sold=("Units_Sold", "sum"),
        Promotion_Rate=("Promotion", lambda s: (s == "Yes").mean()),
    ).reset_index().sort_values("Date")
    return daily


def _engineer_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Lag / rolling / calendar features. Every feature uses only past data relative to its row."""
    d = daily.copy()
    d["dayofweek"] = d["Date"].dt.dayofweek
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)
    d["lag_1"] = d["Units_Sold"].shift(1)
    d["lag_7"] = d["Units_Sold"].shift(7)
    d["roll_7_mean"] = d["Units_Sold"].shift(1).rolling(7).mean()
    d["roll_14_mean"] = d["Units_Sold"].shift(1).rolling(14).mean()
    d["roll_7_std"] = d["Units_Sold"].shift(1).rolling(7).std()
    return d


FEATURE_COLS = ["dayofweek", "is_weekend", "lag_1", "lag_7", "roll_7_mean", "roll_14_mean", "roll_7_std", "Promotion_Rate"]


def _baseline_predict(d: pd.DataFrame) -> pd.Series:
    """Same baseline logic as forecasting.py (7-day trailing average), for apples-to-apples comparison."""
    return d["roll_7_mean"]


def train_and_backtest(df: pd.DataFrame, sku: str) -> dict:
    """
    Train a RandomForestRegressor for one SKU (network-wide daily total),
    chronologically backtest it against the baseline on the last
    TEST_WINDOW_DAYS days, and return metrics + predictions for both.
    """
    daily = _network_daily_series(df, sku)
    feat = _engineer_features(daily).dropna().reset_index(drop=True)

    if len(feat) < TEST_WINDOW_DAYS + 30:
        return {"error": "Not enough history to train and backtest reliably for this SKU."}

    split_idx = len(feat) - TEST_WINDOW_DAYS
    train, test = feat.iloc[:split_idx], feat.iloc[split_idx:]

    X_train, y_train = train[FEATURE_COLS], train["Units_Sold"]
    X_test, y_test = test[FEATURE_COLS], test["Units_Sold"]

    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    rf_pred_test = model.predict(X_test)
    baseline_pred_test = _baseline_predict(test).values

    def _metrics(actual, pred):
        actual, pred = np.array(actual), np.array(pred)
        mae = float(np.mean(np.abs(actual - pred)))
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        mape = float(np.mean(np.abs((actual - pred) / np.where(actual == 0, np.nan, actual))) * 100)
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    rf_metrics = _metrics(y_test, rf_pred_test)
    baseline_metrics = _metrics(y_test, baseline_pred_test)

    importances = dict(zip(FEATURE_COLS, model.feature_importances_.round(4).tolist()))

    return {
        "error": None,
        "test_dates": test["Date"].tolist(),
        "test_actual": y_test.tolist(),
        "rf_pred": rf_pred_test.tolist(),
        "baseline_pred": baseline_pred_test.tolist(),
        "rf_metrics": rf_metrics,
        "baseline_metrics": baseline_metrics,
        "feature_importances": importances,
        "model": model,
        "full_features": feat,  # needed for recursive future forecasting
    }


def forecast_forward(df: pd.DataFrame, sku: str, model_bundle: dict, as_of_date: pd.Timestamp, forecast_days: int) -> dict:
    """
    Recursive multi-step forecast beyond as_of_date using the trained RF model:
    each predicted day's Units_Sold feeds into the lag/rolling features for the
    next day's prediction, since those features aren't known in advance.
    """
    if model_bundle.get("error"):
        return {"forecast_dates": [], "forecast_values": []}

    model = model_bundle["model"]
    history = model_bundle["full_features"][["Date", "Units_Sold", "Promotion_Rate"]].copy()
    history = history[history["Date"] <= as_of_date].reset_index(drop=True)

    values = list(history["Units_Sold"])
    dates = list(history["Date"])
    avg_promo_rate = float(history["Promotion_Rate"].tail(30).mean())

    forecast_dates, forecast_values = [], []
    for step in range(forecast_days):
        next_date = dates[-1] + pd.Timedelta(days=1)
        dayofweek = next_date.dayofweek
        is_weekend = int(dayofweek >= 5)
        lag_1 = values[-1]
        lag_7 = values[-7] if len(values) >= 7 else values[-1]
        roll_7_mean = float(np.mean(values[-7:]))
        roll_14_mean = float(np.mean(values[-14:])) if len(values) >= 14 else roll_7_mean
        roll_7_std = float(np.std(values[-7:])) if len(values) >= 2 else 0.0

        x = pd.DataFrame([{
            "dayofweek": dayofweek, "is_weekend": is_weekend, "lag_1": lag_1, "lag_7": lag_7,
            "roll_7_mean": roll_7_mean, "roll_14_mean": roll_14_mean, "roll_7_std": roll_7_std,
            "Promotion_Rate": avg_promo_rate,
        }])[FEATURE_COLS]
        pred = max(0.0, float(model.predict(x)[0]))

        forecast_dates.append(next_date)
        forecast_values.append(pred)
        dates.append(next_date)
        values.append(pred)

    return {"forecast_dates": forecast_dates, "forecast_values": forecast_values}
