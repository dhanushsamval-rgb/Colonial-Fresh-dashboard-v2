"""
inventory_analysis.py
----------------------
Movement-class analysis, inventory snapshots, lead-time tracking, expiry
tracking, demand-based reorder calculation and risk classification.

All "current" figures are anchored to `as_of_date` (the most recent date in
the dataset), since this is historical data being used to demonstrate a
real-time-style dashboard, not a live system. This is clearly labelled in
the app UI wherever these values are shown.
"""

import pandas as pd
from forecasting import baseline_moving_average_forecast

SAFETY_STOCK_RATE = 0.20  # 20% of forecasted demand during lead time
EXPIRY_RISK_DAYS_THRESHOLD = 2  # days remaining <= this => Expiry Risk
OVERSTOCK_MULTIPLIER = 2.0  # stock > this x lead-time demand => Overstock Risk
RECENT_WINDOW_DAYS = 30  # window used for "recent wastage / stockout" KPIs


def movement_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average daily Units_Sold per product across the FULL dataset,
    independently of the built-in Movement_Class label, so the fast/moderate
    /slow classification shown to the user is demonstrably data-driven.
    """
    stats = (
        df.groupby(["Product_ID", "Product_Name"])["Units_Sold"]
        .agg(Avg_Daily_Units_Sold="mean", Max_Daily_Units_Sold="max", Min_Daily_Units_Sold="min")
        .reset_index()
    )
    stats = stats.sort_values("Avg_Daily_Units_Sold", ascending=False).reset_index(drop=True)

    # Derive a movement label purely from the calculated averages (rank-based),
    # rather than assuming the dataset's own label is correct.
    labels = ["Fast-moving", "Moderate-moving", "Slow-moving"]
    stats["Calculated_Movement_Class"] = labels[: len(stats)]
    return stats


def current_snapshot(df: pd.DataFrame, store: str, product_id: str, as_of_date: pd.Timestamp) -> dict:
    """Most recent Opening/Incoming/Sold/Closing/Wastage/Stockout for one Store+SKU."""
    row = df[
        (df["Store"] == store) & (df["Product_ID"] == product_id) & (df["Date"] == as_of_date)
    ]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "Opening_Stock": int(r["Opening_Stock"]),
        "Incoming_Stock": int(r["Incoming_Stock"]),
        "Units_Sold": int(r["Units_Sold"]),
        "Closing_Stock": int(r["Closing_Stock"]),
        "Wastage": int(r["Wastage"]),
        "Stockout": r["Stockout"],
    }


def estimated_pipeline_incoming_stock(
    df: pd.DataFrame, store: str, product_id: str, as_of_date: pd.Timestamp, lead_time_days: int
) -> float:
    """
    SIMULATED ESTIMATE: the dataset does not track future/pending purchase
    orders explicitly, only the Incoming_Stock that arrived on each historical
    day. As a reasonable Sprint 2 proxy for "stock currently in the pipeline",
    we average the Incoming_Stock actually received over the last
    `lead_time_days` days. This is clearly labelled as an estimate in the UI.
    """
    mask = (
        (df["Store"] == store)
        & (df["Product_ID"] == product_id)
        & (df["Date"] <= as_of_date)
        & (df["Date"] > as_of_date - pd.Timedelta(days=lead_time_days))
    )
    window = df.loc[mask, "Incoming_Stock"]
    if window.empty:
        return 0.0
    return float(window.mean())


def lead_time_metrics(
    df: pd.DataFrame, store: str, product_id: str, as_of_date: pd.Timestamp, avg_daily_forecast: float
) -> dict:
    row = df[(df["Store"] == store) & (df["Product_ID"] == product_id)].iloc[0]
    lead_time_days = int(row["Lead_Time_Days"])
    forecast_during_lead_time = avg_daily_forecast * lead_time_days

    snap = current_snapshot(df, store, product_id, as_of_date)
    current_stock = snap.get("Closing_Stock", 0)
    incoming_stock_est = estimated_pipeline_incoming_stock(df, store, product_id, as_of_date, lead_time_days)

    return {
        "Lead_Time_Days": lead_time_days,
        "Forecasted_Demand_During_Lead_Time": forecast_during_lead_time,
        "Current_Stock": current_stock,
        "Incoming_Stock_Estimate": incoming_stock_est,
    }


def expiry_metrics(df: pd.DataFrame, store: str, product_id: str, as_of_date: pd.Timestamp) -> dict:
    """
    Expiry tracking for the most recently received stock lot as of
    `as_of_date`. LIMITATION (disclosed in-app): this mock dataset records
    aggregate daily stock rather than individual batches, so Days_Remaining
    reflects the shelf life of the freshest lot on hand, not a full batch-
    level FEFO breakdown. Clearly labelled as simulated/prototype.
    """
    row = df[
        (df["Store"] == store) & (df["Product_ID"] == product_id) & (df["Date"] == as_of_date)
    ]
    if row.empty:
        return {}
    r = row.iloc[0]
    shelf_life = int(r["Shelf_Life_Days"])
    expiry_date = pd.Timestamp(r["Expiry_Date"])
    days_remaining = (expiry_date - as_of_date).days
    return {
        "Shelf_Life_Days": shelf_life,
        "Expiry_Date": expiry_date,
        "Days_Remaining": days_remaining,
    }


def calculate_reorder(
    forecasted_demand_during_lead_time: float, current_stock: float, incoming_stock: float
) -> dict:
    """
    Recommended Order = Forecasted Demand During Lead Time + Safety Stock
                         - Current Stock - Incoming Stock
    Safety Stock = 20% of Forecasted Demand During Lead Time.
    Negative results are floored at 0.
    """
    safety_stock = SAFETY_STOCK_RATE * forecasted_demand_during_lead_time
    raw_order = forecasted_demand_during_lead_time + safety_stock - current_stock - incoming_stock
    recommended_order = max(0, round(raw_order))
    return {
        "Safety_Stock": safety_stock,
        "Raw_Calculation": raw_order,
        "Recommended_Order": recommended_order,
    }


def classify_risk(
    current_stock: float,
    forecasted_demand_during_lead_time: float,
    days_remaining: int,
    overstock_threshold_multiplier: float = OVERSTOCK_MULTIPLIER,
) -> str:
    """
    Priority: Expiry Risk > Stockout Risk > Overstock Risk > Normal.

    - Expiry Risk: days remaining on the freshest lot <= threshold AND stock > 0
    - Stockout Risk: current stock cannot cover forecasted demand during lead time
    - Overstock Risk: current stock far exceeds forecasted lead-time demand
    - Normal: none of the above
    """
    if current_stock > 0 and days_remaining <= EXPIRY_RISK_DAYS_THRESHOLD:
        return "Expiry Risk"
    if current_stock < forecasted_demand_during_lead_time:
        return "Stockout Risk"
    if forecasted_demand_during_lead_time > 0 and current_stock > overstock_threshold_multiplier * forecasted_demand_during_lead_time:
        return "Overstock Risk"
    return "Normal"


def assess_stock_position(
    current_stock: float, forecast_during_lead_time: float, safety_stock: float,
    overstock_multiplier: float = OVERSTOCK_MULTIPLIER,
) -> dict:
    """
    Turn the same numbers already used for the reorder calculation into a plain-language
    verdict on whether a given stock level is right, understocked, or overstocked -
    grounded entirely in the demand forecast and the reorder-point logic already used
    everywhere else in the app (no separate/second forecast, no new assumptions).

    Bands (all derived from the existing reorder formula's own building blocks):
    - < Forecast During Lead Time                          -> Understocked (can't even cover the bare minimum)
    - Forecast During Lead Time .. Reorder Point            -> Below Target (survives, but under the 20% safety buffer)
    - Reorder Point .. Overstock Threshold                  -> Well Stocked (right in the healthy zone)
    - > Overstock Threshold (overstock_multiplier x FDLT)   -> Overstocked (tying up cash, raising expiry risk)
    """
    reorder_point = forecast_during_lead_time + safety_stock  # = the target level a fresh order should bring stock to
    overstock_threshold = overstock_multiplier * forecast_during_lead_time if forecast_during_lead_time > 0 else None

    if forecast_during_lead_time <= 0:
        status = "Insufficient Data"
        message = "No meaningful demand forecast is available for this combination yet, so a stocking verdict can't be calculated."
    elif current_stock < forecast_during_lead_time:
        status = "Understocked"
        message = (
            f"Current stock ({current_stock:.0f} units) is below the {forecast_during_lead_time:.0f} units "
            f"the demand forecast says are needed just to survive until the next delivery arrives. "
            f"This is a genuine stockout risk, not just a thin buffer."
        )
    elif current_stock < reorder_point:
        status = "Below Target"
        message = (
            f"Current stock ({current_stock:.0f} units) covers forecasted demand during the lead time "
            f"({forecast_during_lead_time:.0f} units) but sits below the {reorder_point:.0f}-unit reorder point "
            f"(forecast + 20% safety stock). Workable, but with little margin for a demand spike."
        )
    elif overstock_threshold is not None and current_stock > overstock_threshold:
        status = "Overstocked"
        message = (
            f"Current stock ({current_stock:.0f} units) is well above the {overstock_threshold:.0f}-unit "
            f"overstock threshold (2x forecasted demand during the lead time). This ties up cash and, for "
            f"a perishable product, raises the chance some of it expires before it sells."
        )
    else:
        status = "Well Stocked"
        message = (
            f"Current stock ({current_stock:.0f} units) sits comfortably between the {forecast_during_lead_time:.0f}-unit "
            f"minimum and the {overstock_threshold:.0f}-unit overstock threshold — right in the healthy zone the "
            f"demand forecast and reorder-point logic are aiming for."
        )

    return {
        "status": status,
        "message": message,
        "forecast_during_lead_time": forecast_during_lead_time,
        "reorder_point": reorder_point,
        "overstock_threshold": overstock_threshold,
        "current_stock": current_stock,
    }


def build_full_snapshot_table(
    df: pd.DataFrame, stores: list, products: pd.DataFrame, as_of_date: pd.Timestamp, forecast_days: int,
    stock_adjustments: dict = None,
) -> pd.DataFrame:
    """
    Build ONE comprehensive table covering every Store x SKU combination, with every
    figure needed across the Forecasting / Inventory / Lead-Time / Expiry / Reorder
    views: today's Opening/Incoming/Sold/Closing/Wastage/Stockout, the forecast, the
    full lead-time + safety-stock + reorder calculation breakdown, and expiry info.

    This exists so the dashboard can show "everything, all stores, all SKUs" at once
    (no manual filtering needed) without duplicating the underlying business logic -
    it reuses the exact same functions as the rest of the app.

    `stock_adjustments`, if given, is a {(store, product_id): net_units} dict of
    manually logged stock receipts (from the Stock Intake tab). These are ADDED to
    the dataset's Current_Stock (Closing_Stock) figure before any downstream
    lead-time, reorder or risk calculation, so a manual stock entry genuinely
    changes what the rest of the app recommends - it isn't just a display note.
    """
    stock_adjustments = stock_adjustments or {}
    rows = []
    for store in stores:
        for _, prod in products.iterrows():
            product_id = prod["Product_ID"]

            snap = current_snapshot(df, store, product_id, as_of_date)
            fc = baseline_moving_average_forecast(df, store, product_id, as_of_date, forecast_days)
            avg_daily = fc["avg_daily_rate"]

            lt = lead_time_metrics(df, store, product_id, as_of_date, avg_daily)
            adjustment = stock_adjustments.get((store, product_id), 0)
            lt["Current_Stock"] = lt["Current_Stock"] + adjustment

            exp = expiry_metrics(df, store, product_id, as_of_date)
            reorder = calculate_reorder(
                lt["Forecasted_Demand_During_Lead_Time"], lt["Current_Stock"], lt["Incoming_Stock_Estimate"]
            )
            risk = classify_risk(
                lt["Current_Stock"], lt["Forecasted_Demand_During_Lead_Time"], exp.get("Days_Remaining", 999)
            )

            rows.append({
                "Store": store,
                "SKU": product_id,
                "Product": prod["Product_Name"],
                "Movement": prod["Movement_Class"],
                # Today's inventory snapshot
                "Opening_Stock": snap.get("Opening_Stock"),
                "Incoming_Stock_Today": snap.get("Incoming_Stock"),
                "Units_Sold_Today": snap.get("Units_Sold"),
                "Current_Stock": lt["Current_Stock"],  # = today's Closing_Stock + manual adjustments
                "Manual_Adjustment": adjustment,
                "Wastage_Today": snap.get("Wastage"),
                "Stockout_Today": snap.get("Stockout"),
                # Forecast
                "Avg_Daily_Forecast": round(avg_daily, 2),
                "Forecast_Period_Demand": round(avg_daily * forecast_days, 1),
                # Lead time
                "Lead_Time_Days": lt["Lead_Time_Days"],
                "Forecast_During_Lead_Time": round(lt["Forecasted_Demand_During_Lead_Time"], 1),
                "Incoming_Stock_Estimate": round(lt["Incoming_Stock_Estimate"], 1),
                # Reorder calculation (full breakdown)
                "Safety_Stock": round(reorder["Safety_Stock"], 1),
                "Raw_Calculation": round(reorder["Raw_Calculation"], 1),
                "Recommended_Order": reorder["Recommended_Order"],
                # Expiry
                "Shelf_Life_Days": exp.get("Shelf_Life_Days"),
                "Expiry_Date": exp.get("Expiry_Date").date() if exp.get("Expiry_Date") is not None else None,
                "Days_To_Expiry": exp.get("Days_Remaining"),
                # Risk
                "Risk": risk,
            })
    return pd.DataFrame(rows)


def build_risk_table(
    df: pd.DataFrame, stores: list, products: pd.DataFrame, as_of_date: pd.Timestamp, forecast_days: int,
    stock_adjustments: dict = None,
) -> pd.DataFrame:
    """
    Build the full Risk Dashboard table:
    SKU | Movement | Store | Current Stock | Forecast | Lead Time | Expiry | Risk | Recommended Order
    Populated dynamically for every store x SKU combination in scope. `stock_adjustments`
    behaves exactly as in build_full_snapshot_table.
    """
    stock_adjustments = stock_adjustments or {}
    rows = []
    for store in stores:
        for _, prod in products.iterrows():
            product_id = prod["Product_ID"]
            fc = baseline_moving_average_forecast(df, store, product_id, as_of_date, forecast_days)
            avg_daily = fc["avg_daily_rate"]

            lt = lead_time_metrics(df, store, product_id, as_of_date, avg_daily)
            lt["Current_Stock"] = lt["Current_Stock"] + stock_adjustments.get((store, product_id), 0)

            exp = expiry_metrics(df, store, product_id, as_of_date)
            reorder = calculate_reorder(
                lt["Forecasted_Demand_During_Lead_Time"], lt["Current_Stock"], lt["Incoming_Stock_Estimate"]
            )
            risk = classify_risk(
                lt["Current_Stock"], lt["Forecasted_Demand_During_Lead_Time"], exp.get("Days_Remaining", 999)
            )

            rows.append(
                {
                    "Store": store,
                    "SKU": product_id,
                    "Product": prod["Product_Name"],
                    "Movement": prod["Movement_Class"],
                    "Current_Stock": lt["Current_Stock"],
                    "Forecast_Period_Demand": round(avg_daily * forecast_days, 1),
                    "Avg_Daily_Forecast": round(avg_daily, 2),
                    "Lead_Time_Days": lt["Lead_Time_Days"],
                    "Forecast_During_Lead_Time": round(lt["Forecasted_Demand_During_Lead_Time"], 1),
                    "Days_To_Expiry": exp.get("Days_Remaining"),
                    "Risk": risk,
                    "Recommended_Order": reorder["Recommended_Order"],
                }
            )
    return pd.DataFrame(rows)
