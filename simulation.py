"""
simulation.py
--------------
"Live Simulation" engine for the Colonial Fresh prototype.

This module generates a plausible SYNTHETIC continuation of daily operations
(day-by-day) for all Store x SKU combinations, starting the day after the
real historical dataset ends. It is explicitly a SIMULATION, not real data:

- Each simulated day's demand is generated from patterns learned from the
  real 2024-2025 history for that Store+SKU (recent average daily rate,
  day-of-week seasonality, promotion effect size) plus random day-to-day
  variation. It is not a prediction of real future sales.
- The same business logic used elsewhere in the app for forecasting
  (7-day moving average), lead-time-aware reorder calculation and risk
  classification is reused here (imported directly from forecasting.py /
  inventory_analysis.py) so results stay consistent between the historical
  dashboard and the live simulation.
- Reorders recommended by the system are automatically "placed" and arrive
  after the SKU's Lead_Time_Days, closing the demand -> forecast -> reorder
  -> restock loop so it can be watched running live.

This is a Sprint 2/3 conceptual demonstration of the proposed decision logic
in motion - not a validated demand-forecasting model and not connected to
any live point-of-sale or supplier system.
"""

from collections import deque
import numpy as np
import pandas as pd

from forecasting import MOVING_AVERAGE_WINDOW
from inventory_analysis import calculate_reorder, classify_risk

RECENT_SALES_BUFFER = 14          # how many recent daily sales values we keep per combo
DAY_TO_DAY_NOISE_STD = 0.15        # +/-15% random variation around the base rate
WEEKDAY_FACTOR_CLIP = (0.5, 1.6)   # sanity bounds on learned seasonality
PROMO_BOOST_CLIP = (1.0, 2.0)      # sanity bounds on learned promotion uplift
DEFAULT_PROMO_BOOST = 1.15
LEAD_TIME_VARIABILITY_RATE = 0.3   # SIMULATED: assumed std as a fraction of the dataset's
                                    # fixed Lead_Time_Days. The real dataset records a single
                                    # constant lead time per SKU with no day-to-day variability
                                    # to learn from, so this spread is an explicit, disclosed
                                    # assumption used only to give the Live Simulation and the
                                    # Lead-Time Anticipation tab something genuine to observe
                                    # and learn from, rather than a constant that can't be
                                    # "anticipated" at all.

SIM_LOG_COLUMNS = [
    "Date", "Store", "Product_ID", "Product_Name", "Opening_Stock",
    "Incoming_Stock", "Units_Sold", "Closing_Stock", "Wastage", "Stockout",
    "Promotion", "Avg_Daily_Forecast", "Forecast_During_Lead_Time",
    "Recommended_Order", "Risk",
]

LEAD_TIME_LOG_COLUMNS = ["Order_Date", "Store", "Product_ID", "Product_Name", "Planned_Lead_Time", "Actual_Lead_Time", "Arrival_Date"]


def _learn_combo_profile(hist: pd.DataFrame) -> dict:
    """
    Learn a lightweight generative profile for one Store+SKU combo from its
    real historical rows: recent sales buffer, day-of-week seasonality,
    promotion probability/effect, wastage rate, lead time and shelf life.
    """
    hist = hist.sort_values("Date")
    overall_mean = hist["Units_Sold"].mean() or 1.0

    weekday_means = hist.groupby(hist["Date"].dt.dayofweek)["Units_Sold"].mean()
    weekday_factors = (weekday_means / overall_mean).clip(*WEEKDAY_FACTOR_CLIP).to_dict()
    # Fill any missing weekday (shouldn't happen with 2 years of data, but be safe)
    weekday_factors = {d: weekday_factors.get(d, 1.0) for d in range(7)}

    promo_mask = hist["Promotion"] == "Yes"
    promo_probability = float(promo_mask.mean())
    if promo_mask.sum() > 0 and (~promo_mask).sum() > 0:
        promo_mean = hist.loc[promo_mask, "Units_Sold"].mean()
        non_promo_mean = hist.loc[~promo_mask, "Units_Sold"].mean()
        promo_boost = (promo_mean / non_promo_mean) if non_promo_mean > 0 else DEFAULT_PROMO_BOOST
        promo_boost = float(np.clip(promo_boost, *PROMO_BOOST_CLIP))
    else:
        promo_boost = DEFAULT_PROMO_BOOST

    wastage_lambda = float(hist["Wastage"].mean())
    recent_sales = deque(hist["Units_Sold"].tail(RECENT_SALES_BUFFER).tolist(), maxlen=RECENT_SALES_BUFFER)

    last_row = hist.iloc[-1]
    return {
        "recent_sales": recent_sales,
        "weekday_factors": weekday_factors,
        "promo_probability": promo_probability,
        "promo_boost": promo_boost,
        "wastage_lambda": max(wastage_lambda, 0.0),
        "lead_time_days": int(last_row["Lead_Time_Days"]),
        "shelf_life_days": int(last_row["Shelf_Life_Days"]),
        "current_stock": float(last_row["Closing_Stock"]),
        "pending_orders": [],  # list of (arrival_date, quantity)
        "product_name": last_row["Product_Name"],
    }


def init_simulation_state(df: pd.DataFrame, stores: list, products: pd.DataFrame, as_of_date: pd.Timestamp, seed: int = None) -> dict:
    """Build the initial simulation state, anchored at the real dataset's last day."""
    rng = np.random.default_rng(seed)
    combos = {}
    for store in stores:
        for _, prod in products.iterrows():
            sku = prod["Product_ID"]
            hist = df[(df["Store"] == store) & (df["Product_ID"] == sku)]
            if hist.empty:
                continue
            combos[(store, sku)] = _learn_combo_profile(hist)

    return {
        "sim_date": as_of_date,
        "combos": combos,
        "log": pd.DataFrame(columns=SIM_LOG_COLUMNS),
        "lead_time_log": pd.DataFrame(columns=LEAD_TIME_LOG_COLUMNS),
        "rng": rng,
        "days_simulated": 0,
    }


def simulate_next_day(state: dict) -> dict:
    """
    Advance the simulation by exactly one day across every Store x SKU combo.
    Mutates and returns `state`.
    """
    rng = state["rng"]
    new_date = state["sim_date"] + pd.Timedelta(days=1)
    weekday = new_date.dayofweek

    new_rows = []
    new_lead_time_rows = []
    for (store, sku), c in state["combos"].items():
        # 1) Resolve any pending orders arriving today
        arriving_today = sum(qty for (arr_date, qty) in c["pending_orders"] if arr_date == new_date)
        c["pending_orders"] = [(arr_date, qty) for (arr_date, qty) in c["pending_orders"] if arr_date != new_date]

        opening_stock = c["current_stock"] + arriving_today

        # 2) Generate today's demand from the learned profile
        base_rate = float(np.mean(c["recent_sales"])) if c["recent_sales"] else 1.0
        weekday_factor = c["weekday_factors"].get(weekday, 1.0)
        promo_today = rng.random() < c["promo_probability"]
        promo_multiplier = c["promo_boost"] if promo_today else 1.0
        noise = rng.normal(1.0, DAY_TO_DAY_NOISE_STD)
        raw_demand = base_rate * weekday_factor * promo_multiplier * noise
        demand = max(0, round(raw_demand))

        # 3) Sales are capped by what's actually on the shelf (lost sales, no backorders)
        units_sold = min(demand, opening_stock)
        stockout = "Yes" if demand > opening_stock else "No"

        # 4) Wastage: random variation around this combo's historical average daily wastage,
        #    capped so it can never exceed what's left after today's sales
        remaining_after_sale = opening_stock - units_sold
        wastage = min(remaining_after_sale, rng.poisson(c["wastage_lambda"]))
        wastage = max(0, wastage)

        closing_stock = remaining_after_sale - wastage

        # 5) Update rolling sales buffer used for the forecast
        c["recent_sales"].append(units_sold)
        window_vals = list(c["recent_sales"])[-MOVING_AVERAGE_WINDOW:]
        avg_daily_forecast = float(np.mean(window_vals)) if window_vals else 0.0

        # 6) Lead-time-aware demand-based reorder (reuses the same function as the historical dashboard)
        lead_time_days = c["lead_time_days"]
        forecast_during_lead_time = avg_daily_forecast * lead_time_days
        incoming_pipeline = sum(qty for (_, qty) in c["pending_orders"])
        reorder = calculate_reorder(forecast_during_lead_time, closing_stock, incoming_pipeline)
        recommended_order = reorder["Recommended_Order"]

        if recommended_order > 0:
            planned_lead_time = lead_time_days
            lead_time_std = max(1.0, planned_lead_time * LEAD_TIME_VARIABILITY_RATE)
            actual_lead_time = int(max(1, round(rng.normal(planned_lead_time, lead_time_std))))
            arrival_date = new_date + pd.Timedelta(days=actual_lead_time)
            c["pending_orders"].append((arrival_date, recommended_order))
            new_lead_time_rows.append({
                "Order_Date": new_date, "Store": store, "Product_ID": sku, "Product_Name": c["product_name"],
                "Planned_Lead_Time": planned_lead_time, "Actual_Lead_Time": actual_lead_time,
                "Arrival_Date": arrival_date,
            })

        # 7) Risk classification (reuses the same classify_risk thresholds/logic as the
        #    historical dashboard - unchanged). For the LIVE simulation only, we feed in
        #    "stock position" (on-hand + already-ordered/in-transit) rather than raw
        #    on-hand stock, because a fresh order is genuinely already inbound every
        #    review cycle here. Using raw on-hand stock would flag "Stockout Risk" on
        #    almost every day purely as an artifact of the tight 1-2 day lead times in a
        #    continuously-reviewed loop, even when deliveries are arriving on schedule and
        #    no customer demand is actually going unmet. Days-to-expiry keeps the same
        #    simplification used in the historical view (freshest-lot shelf life).
        stock_position = closing_stock + incoming_pipeline
        risk = classify_risk(stock_position, forecast_during_lead_time, c["shelf_life_days"])

        c["current_stock"] = closing_stock

        new_rows.append({
            "Date": new_date, "Store": store, "Product_ID": sku, "Product_Name": c["product_name"],
            "Opening_Stock": opening_stock, "Incoming_Stock": arriving_today, "Units_Sold": units_sold,
            "Closing_Stock": closing_stock, "Wastage": wastage, "Stockout": stockout,
            "Promotion": "Yes" if promo_today else "No",
            "Avg_Daily_Forecast": round(avg_daily_forecast, 2),
            "Forecast_During_Lead_Time": round(forecast_during_lead_time, 2),
            "Recommended_Order": recommended_order, "Risk": risk,
        })

    state["sim_date"] = new_date
    state["days_simulated"] += 1
    if new_rows:
        state["log"] = pd.concat([state["log"], pd.DataFrame(new_rows)], ignore_index=True)
    if new_lead_time_rows:
        state["lead_time_log"] = pd.concat([state["lead_time_log"], pd.DataFrame(new_lead_time_rows)], ignore_index=True)
    return state


def anticipate_lead_time(lead_time_log: pd.DataFrame, store: str, sku: str, planned_lead_time: int) -> dict:
    """
    Turn the observed actual-vs-planned delivery history for one Store+SKU (built up as
    the Live Simulation runs) into an "anticipated" next delivery estimate: the observed
    mean actual lead time and its standard deviation, giving a genuine data-driven
    prediction range rather than just repeating the fixed textbook lead time.

    Returns None-valued fields when no observations exist yet for this combo (i.e. the
    simulation hasn't placed any orders for it), so the caller can fall back to the
    static planned lead time and say so honestly.
    """
    sub = lead_time_log[(lead_time_log["Store"] == store) & (lead_time_log["Product_ID"] == sku)]
    n = len(sub)
    if n == 0:
        return {
            "n_observations": 0, "anticipated_days": planned_lead_time, "std_days": None,
            "low": planned_lead_time, "high": planned_lead_time,
        }
    actual = sub["Actual_Lead_Time"].astype(float)
    mean = float(actual.mean())
    std = float(actual.std()) if n > 1 else 0.0
    return {
        "n_observations": n,
        "anticipated_days": round(mean, 1),
        "std_days": round(std, 1),
        "low": max(1, round(mean - std)),
        "high": round(mean + std),
    }
