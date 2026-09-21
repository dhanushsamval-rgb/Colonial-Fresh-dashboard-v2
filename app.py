"""
app.py
------
AI-Based Demand Forecasting and Inventory Management System - Colonial Fresh
MGT5ELS Assessment 3 - Sprint 2 (Conceptualisation & Design) prototype.

Run with:  streamlit run app.py

This is a Sprint 2 functional DEMONSTRATION of the proposed design, built on
a fully SIMULATED dataset (2024-2025). It is not connected to any live
Colonial Fresh system. No business-improvement results are claimed - only
the mechanics of the proposed decision-support logic are demonstrated.
"""

import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_utils import load_and_prepare, get_stores, get_products, get_as_of_date
from forecasting import baseline_moving_average_forecast, linear_trend_forecast
from inventory_analysis import (
    movement_analysis,
    build_risk_table,
    build_full_snapshot_table,
    assess_stock_position,
    RECENT_WINDOW_DAYS,
)
import simulation
import ml_forecast

st.set_page_config(page_title="Colonial Fresh | Demand & Inventory Prototype", layout="wide")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
try:
    df, load_messages = load_and_prepare()
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

STORES = get_stores(df)
PRODUCTS = get_products(df)
AS_OF_DATE = get_as_of_date(df)

# ---------------------------------------------------------------------------
# Header / disclaimer banner
# ---------------------------------------------------------------------------
st.title("🥑 Colonial Fresh — AI-Based Demand Forecasting & Inventory Management")
st.caption("MGT5ELS Assessment 3 · Sprint 2 Prototype — Conceptualisation & Design")

st.info(
    f"**Prototype notice:** This dashboard runs on a fully **simulated** dataset "
    f"({df['Date'].min().date()} to {df['Date'].max().date()}) built for academic purposes. "
    f"It is **not** real Colonial Fresh data and is **not** connected to any live system. "
    f"The most recent date in the dataset ({AS_OF_DATE.date()}) is treated as **'today'** "
    f"for all current-inventory, lead-time and expiry calculations, to demonstrate how a "
    f"real-time dashboard would behave. Live system integration is proposed as future work "
    f"beyond this Sprint 2 design.",
    icon="ℹ️",
)

if load_messages:
    with st.expander("Data validation & cleaning log"):
        for m in load_messages:
            st.write("•", m)

# ---------------------------------------------------------------------------
# Scope: this dashboard always shows ALL stores and ALL SKUs together (no
# filtering), so it can be presented end-to-end without pausing to change
# dropdowns. The only control is the forecast horizon, since that's a genuine
# "what-if" parameter rather than something that hides data.
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
forecast_period = st.sidebar.selectbox("Forecast period (days)", [7, 14, 30], index=0)
st.sidebar.caption("Every tab below always shows **all 6 stores × all 3 SKUs** together — nothing is filtered or hidden.")

stores_selected = STORES
product_ids_selected = PRODUCTS["Product_ID"].tolist()
filtered_df = df
products_selected_df = PRODUCTS

# ---------------------------------------------------------------------------
# Stock Intake ledger (session-only): manually logged stock receipts that
# genuinely shift Current_Stock, and therefore every downstream lead-time,
# reorder and risk calculation across the whole app - not just a display note.
# ---------------------------------------------------------------------------
if "stock_ledger" not in st.session_state:
    st.session_state.stock_ledger = []  # list of dicts: Date, Store, SKU, Product, Quantity, Note, Logged_At

def _stock_adjustments_dict():
    adj = {}
    for entry in st.session_state.stock_ledger:
        key = (entry["Store"], entry["SKU"])
        adj[key] = adj.get(key, 0) + entry["Quantity"]
    return adj

stock_adjustments = _stock_adjustments_dict()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_movement, tab_forecast, tab_ml, tab_inventory, tab_intake, tab_leadtime, \
    tab_leadtime_ai, tab_expiry, tab_reorder, tab_risk, tab_live, tab_architecture = st.tabs(
        [
            "1️⃣ Overview", "2️⃣ Movement Analysis", "3️⃣ Demand Forecasting", "🤖 ML Forecast",
            "4️⃣ Inventory Monitoring", "📥 Stock Intake", "5️⃣ Lead-Time Tracking", "🚚 Lead-Time Anticipation",
            "6️⃣ Expiry Tracking", "7️⃣ Reorder Recommendation", "8️⃣ Risk Dashboard", "🔴 Live Simulation",
            "9️⃣ Solution Architecture",
        ]
    )

# Pre-compute the risk table and the full all-store x all-SKU snapshot once (reused across tabs).
# Both incorporate any manually logged Stock Intake adjustments, so a manual entry genuinely
# changes every recommendation across the app, not just what's displayed on one tab.
risk_table = build_risk_table(df, stores_selected, products_selected_df, AS_OF_DATE, forecast_period, stock_adjustments)
full_table = build_full_snapshot_table(df, STORES, PRODUCTS, AS_OF_DATE, forecast_period, stock_adjustments)


def _risk_style(val):
    colors = {"Normal": "background-color:#e6f4ea", "Stockout Risk": "background-color:#fce8e6",
              "Overstock Risk": "background-color:#fef3e2", "Expiry Risk": "background-color:#f3e8fd"}
    return colors.get(val, "")

# ===========================================================================
# TAB 1: OVERVIEW DASHBOARD
# ===========================================================================
with tab_overview:
    st.subheader("Overview Dashboard")
    st.caption(f"All 6 stores × all 3 SKUs · forecast period = {forecast_period} days · as of {AS_OF_DATE.date()}.")

    total_current_inventory = risk_table["Current_Stock"].sum()
    total_forecast_demand = risk_table["Forecast_Period_Demand"].sum()
    total_incoming = filtered_df[filtered_df["Date"] == AS_OF_DATE]["Incoming_Stock"].sum()
    recent_wastage = filtered_df[
        filtered_df["Date"] > AS_OF_DATE - pd.Timedelta(days=RECENT_WINDOW_DAYS)
    ]["Wastage"].sum()
    stockout_rows_recent = filtered_df[
        (filtered_df["Date"] > AS_OF_DATE - pd.Timedelta(days=RECENT_WINDOW_DAYS))
        & (filtered_df["Stockout"] == "Yes")
    ]
    products_attention = (risk_table["Risk"] != "Normal").sum()
    total_recommended_order = risk_table["Recommended_Order"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Current Inventory (units)", f"{total_current_inventory:,.0f}")
    c2.metric(f"Forecasted Demand (next {forecast_period}d)", f"{total_forecast_demand:,.0f}")
    c3.metric("Incoming Stock (today)", f"{total_incoming:,.0f}")
    c4.metric(f"Wastage (last {RECENT_WINDOW_DAYS}d)", f"{recent_wastage:,.0f}")

    c5, c6, c7 = st.columns(3)
    c5.metric(f"Stockout Occurrences (last {RECENT_WINDOW_DAYS}d)", f"{len(stockout_rows_recent):,}")
    c6.metric("Store × SKU Combos Needing Attention", f"{products_attention} / {len(risk_table)}")
    c7.metric("Total Recommended Reorder Qty", f"{total_recommended_order:,.0f} units")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Products / Stores Requiring Attention**")
        attention_df = risk_table[risk_table["Risk"] != "Normal"][
            ["Store", "SKU", "Product", "Risk", "Current_Stock", "Recommended_Order"]
        ].sort_values("Risk")
        if attention_df.empty:
            st.success("No stockout, overstock or expiry risks detected in the current scope.")
        else:
            st.dataframe(attention_df, use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**Risk Mix (current scope)**")
        risk_counts = risk_table["Risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        fig = px.pie(risk_counts, names="Risk", values="Count", hole=0.45,
                     color="Risk",
                     color_discrete_map={"Normal": "#2ca02c", "Stockout Risk": "#d62728",
                                          "Overstock Risk": "#ff7f0e", "Expiry Risk": "#9467bd"})
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 2: PRODUCT MOVEMENT ANALYSIS
# ===========================================================================
with tab_movement:
    st.subheader("Product Movement Analysis")
    st.write(
        "Movement class is not assumed - it is **calculated** from average daily "
        "`Units_Sold` across the full 2024-2025 history for each SKU."
    )

    mv = movement_analysis(df)
    st.dataframe(
        mv.rename(columns={
            "Avg_Daily_Units_Sold": "Avg Daily Units Sold",
            "Max_Daily_Units_Sold": "Max Daily Units Sold",
            "Min_Daily_Units_Sold": "Min Daily Units Sold",
            "Calculated_Movement_Class": "Calculated Movement Class",
        }),
        use_container_width=True, hide_index=True,
    )

    fig = px.bar(
        mv, x="Product_Name", y="Avg_Daily_Units_Sold", color="Calculated_Movement_Class",
        text_auto=".1f", title="Average Daily Units Sold by Product (2024-2025 full history)",
        labels={"Avg_Daily_Units_Sold": "Avg Daily Units Sold", "Product_Name": "Product"},
        color_discrete_map={"Fast-moving": "#2ca02c", "Moderate-moving": "#ff7f0e", "Slow-moving": "#d62728"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Bananas are confirmed **fast-moving**, Apples **moderate-moving** and "
        "Strawberries **slow-moving** based on calculated average daily sales, "
        "consistent with the three-SKU baseline required by the assessment brief."
    )

# ===========================================================================
# TAB 3: DEMAND FORECASTING — all stores, all SKUs
# ===========================================================================
with tab_forecast:
    st.subheader("Demand Forecasting — All Stores × All SKUs")
    st.write(
        "**Baseline model (primary):** 7-day moving average of historical `Units_Sold`, "
        "projected forward as a flat daily rate. `Simulated_True_Demand` is **not** used "
        "as a model input, to avoid data leakage — it is reserved for later evaluation only. "
        "The chart below aggregates all 6 stores together for each SKU, so all three "
        "products are visible on one screen."
    )

    show_trend = st.checkbox(
        "Also show scikit-learn Linear Regression trend model (comparison only — not claimed superior)",
        value=False,
    )

    sku_colors = {"SKU001": "#2ca02c", "SKU002": "#ff7f0e", "SKU003": "#9467bd"}
    fig = go.Figure()
    for _, prod in PRODUCTS.iterrows():
        sku = prod["Product_ID"]
        # Network-wide historical: sum Units_Sold across all 6 stores, per day, for this SKU
        hist_net = (
            df[df["Product_ID"] == sku].groupby("Date")["Units_Sold"].sum().reset_index().tail(60)
        )
        fig.add_trace(go.Scatter(x=hist_net["Date"], y=hist_net["Units_Sold"], mode="lines",
                                  name=f"{prod['Product_Name']} — Historical (network)",
                                  line=dict(color=sku_colors.get(sku), width=2)))

        # Network-wide forecast: sum each store's baseline forecast for this SKU
        fc_dates, fc_totals = None, None
        for store in STORES:
            fc = baseline_moving_average_forecast(df, store, sku, AS_OF_DATE, forecast_period)
            if fc_dates is None:
                fc_dates = fc["forecast_dates"]
                fc_totals = list(fc["forecast_values"])
            else:
                fc_totals = [a + b for a, b in zip(fc_totals, fc["forecast_values"])]
        fig.add_trace(go.Scatter(x=fc_dates, y=fc_totals, mode="lines+markers",
                                  name=f"{prod['Product_Name']} — Forecast (network)",
                                  line=dict(color=sku_colors.get(sku), dash="dash")))

        if show_trend:
            tr_dates, tr_totals = None, None
            for store in STORES:
                tr = linear_trend_forecast(df, store, sku, AS_OF_DATE, forecast_period)
                if not tr["forecast_dates"]:
                    continue
                if tr_dates is None:
                    tr_dates = tr["forecast_dates"]
                    tr_totals = list(tr["forecast_values"])
                else:
                    tr_totals = [a + b for a, b in zip(tr_totals, tr["forecast_values"])]
            if tr_dates:
                fig.add_trace(go.Scatter(x=tr_dates, y=tr_totals, mode="lines",
                                          name=f"{prod['Product_Name']} — Linear Trend (comparison)",
                                          line=dict(color=sku_colors.get(sku), dash="dot", width=1)))

    fig.update_layout(title="Network-Wide Historical Sales vs Forecast (all stores combined, per SKU)",
                       xaxis_title="Date", yaxis_title="Units Sold (all 6 stores)",
                       legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Forecast summary — every store × SKU**")
    fc_display = full_table[["Store", "Product", "SKU", "Avg_Daily_Forecast", "Forecast_Period_Demand"]].rename(
        columns={"Avg_Daily_Forecast": "Avg Daily Forecast", "Forecast_Period_Demand": f"Forecast (next {forecast_period}d)"}
    )
    st.dataframe(fc_display, use_container_width=True, hide_index=True)

    totals_by_sku = full_table.groupby("Product")["Forecast_Period_Demand"].sum().reset_index()
    cols = st.columns(len(totals_by_sku))
    for col, (_, row) in zip(cols, totals_by_sku.iterrows()):
        col.metric(f"{row['Product']} — Network Total (next {forecast_period}d)", f"{row['Forecast_Period_Demand']:,.0f} units")

    if show_trend:
        st.caption(
            "Linear trend lines are shown for comparison only — no back-testing has been performed "
            "in this Sprint 2 prototype, so no accuracy claim is made about either model."
        )

# ===========================================================================
# TAB: MACHINE LEARNING FORECAST — real, backtested Random Forest per SKU
# ===========================================================================
with tab_ml:
    st.subheader("Machine Learning Forecast — Random Forest (network-wide, per SKU)")
    st.write(
        "This is a genuine machine learning model — scikit-learn `RandomForestRegressor` — trained "
        "on lag, rolling-average and calendar features engineered from historical `Units_Sold`. "
        "Unlike the side-by-side comparison on the Demand Forecasting tab, the numbers below come "
        "from a real **chronological backtest**: the model is trained only on earlier data and "
        f"evaluated on the most recent **{ml_forecast.TEST_WINDOW_DAYS} held-out days** it never saw "
        "during training — so any claim that it beats the baseline is backed by an actual test, not assumed."
    )
    st.caption("`Simulated_True_Demand` is never used as a feature or target, for the same data-leakage reasons as the baseline model.")

    ml_sku_choice = st.selectbox(
        "SKU", PRODUCTS["Product_ID"] + " – " + PRODUCTS["Product_Name"], index=0, key="ml_sku_choice"
    )
    ml_sku_id = ml_sku_choice.split(" – ")[0]

    with st.spinner("Training and backtesting the Random Forest model…"):
        bundle = ml_forecast.train_and_backtest(df, ml_sku_id)

    if bundle.get("error"):
        st.warning(bundle["error"])
    else:
        st.markdown(f"**Backtest — last {ml_forecast.TEST_WINDOW_DAYS} days, network-wide daily totals**")
        m1, m2, m3 = st.columns(3)
        rf_m, bl_m = bundle["rf_metrics"], bundle["baseline_metrics"]
        m1.metric("MAE (lower is better)", f"{rf_m['MAE']:.1f}", delta=f"{rf_m['MAE']-bl_m['MAE']:+.1f} vs baseline", delta_color="inverse")
        m2.metric("RMSE (lower is better)", f"{rf_m['RMSE']:.1f}", delta=f"{rf_m['RMSE']-bl_m['RMSE']:+.1f} vs baseline", delta_color="inverse")
        m3.metric("MAPE (lower is better)", f"{rf_m['MAPE']:.1f}%", delta=f"{rf_m['MAPE']-bl_m['MAPE']:+.1f}pp vs baseline", delta_color="inverse")

        if rf_m["MAE"] < bl_m["MAE"]:
            st.success(
                f"On this backtest, Random Forest beats the 7-day moving-average baseline for "
                f"{ml_sku_id} — {rf_m['MAE']:.1f} vs {bl_m['MAE']:.1f} average units of error per day."
            )
        else:
            st.info(
                f"On this backtest, the simple baseline actually matches or beats Random Forest for "
                f"{ml_sku_id} — {bl_m['MAE']:.1f} vs {rf_m['MAE']:.1f} average units of error per day. "
                f"Reported honestly either way, as required."
            )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bundle["test_dates"], y=bundle["test_actual"], name="Actual", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=bundle["test_dates"], y=bundle["rf_pred"], name="Random Forest", line=dict(color="#2ca02c", dash="dash")))
        fig.add_trace(go.Scatter(x=bundle["test_dates"], y=bundle["baseline_pred"], name="Baseline (7-day MA)", line=dict(color="#ff7f0e", dash="dot")))
        fig.update_layout(title=f"Backtest — Actual vs Predicted, {ml_sku_id} (network-wide)", xaxis_title="Date", yaxis_title="Units Sold", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Feature importances (what the model actually learned from)"):
            imp_df = pd.DataFrame(list(bundle["feature_importances"].items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=False)
            st.dataframe(imp_df, use_container_width=True, hide_index=True)

        st.markdown(f"**Forward forecast — next {forecast_period} days (recursive, network-wide)**")
        fwd = ml_forecast.forecast_forward(df, ml_sku_id, bundle, AS_OF_DATE, forecast_period)
        fwd_df = pd.DataFrame({"Date": [d.date() for d in fwd["forecast_dates"]], "ML Forecast (network)": [round(v, 1) for v in fwd["forecast_values"]]})
        st.dataframe(fwd_df, use_container_width=True, hide_index=True)
        st.caption(
            "Recursive forecast: each predicted day feeds the lag features for the next day, since "
            "those aren't known in advance. Multi-step forecasts like this naturally get less certain "
            "the further out they go — a normal, disclosed limitation of this approach."
        )

# ===========================================================================
# TAB 4: INVENTORY MONITORING — all stores, all SKUs
# ===========================================================================
with tab_inventory:
    st.subheader("Inventory Monitoring — All Stores × All SKUs")
    st.caption(f"As of {AS_OF_DATE.date()} (the most recent date in the dataset).")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Opening Stock", f"{full_table['Opening_Stock'].sum():,.0f}")
    k2.metric("Total Incoming Today", f"{full_table['Incoming_Stock_Today'].sum():,.0f}")
    k3.metric("Total Units Sold Today", f"{full_table['Units_Sold_Today'].sum():,.0f}")
    k4.metric("Total Closing Stock", f"{full_table['Current_Stock'].sum():,.0f}")

    inv_display = full_table[[
        "Store", "Product", "SKU", "Opening_Stock", "Incoming_Stock_Today", "Units_Sold_Today",
        "Current_Stock", "Wastage_Today", "Stockout_Today", "Risk",
    ]].rename(columns={
        "Opening_Stock": "Opening Stock", "Incoming_Stock_Today": "Incoming Stock",
        "Units_Sold_Today": "Units Sold", "Current_Stock": "Closing Stock",
        "Wastage_Today": "Wastage", "Stockout_Today": "Stockout",
    })
    st.dataframe(inv_display.style.map(_risk_style, subset=["Risk"]), use_container_width=True, hide_index=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(full_table, x="Store", y="Current_Stock", color="Product", barmode="group",
                     title="Closing Stock by Store, split by SKU", labels={"Current_Stock": "Closing Stock"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(full_table, x="Store", y="Wastage_Today", color="Product", barmode="group",
                     title="Today's Wastage by Store, split by SKU", labels={"Wastage_Today": "Wastage"})
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB: STOCK INTAKE — manually log stock receipts; feeds every calculation
# ===========================================================================
with tab_intake:
    st.subheader("Stock Intake — Log Received Stock")
    st.write(
        "Log stock as it's physically received. Every entry here is added directly to that "
        "Store × SKU's Current Stock and immediately flows through to Lead-Time, Reorder, "
        "Risk and every other tab in this app — it's a real adjustment, not just a note."
    )

    with st.form("stock_intake_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            intake_store = st.selectbox("Store", STORES)
        with c2:
            intake_product_label = st.selectbox("Product", PRODUCTS["Product_ID"] + " – " + PRODUCTS["Product_Name"])
        with c3:
            intake_qty = st.number_input("Quantity received", min_value=1, step=1, value=50)
        intake_note = st.text_input("Note (optional)", placeholder="e.g. delivery from supplier, manual correction")
        submitted = st.form_submit_button("➕ Log stock receipt")

        if submitted:
            intake_sku = intake_product_label.split(" – ")[0]
            intake_name = PRODUCTS.loc[PRODUCTS["Product_ID"] == intake_sku, "Product_Name"].iloc[0]
            st.session_state.stock_ledger.append({
                "Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "Store": intake_store, "SKU": intake_sku, "Product": intake_name,
                "Quantity": int(intake_qty), "Note": intake_note,
            })
            # Pre-set the AI Assessment widgets' own keyed state directly (not just a plain
            # session_state value) - Streamlit ignores a selectbox's `index=` argument on
            # reruns once the widget has already rendered once, so this is the correct way
            # to make the assessment below jump to whatever was just logged.
            st.session_state["assess_store"] = intake_store
            st.session_state["assess_product"] = intake_product_label
            st.success(f"Logged +{intake_qty} units of {intake_name} at {intake_store}. Every tab now reflects this.")
            st.rerun()

    st.divider()
    st.markdown("### 🤖 AI Stock Assessment")
    st.write(
        "Is the stock level right, understocked, or overstocked? This reads the same demand "
        "forecast and reorder-point numbers already calculated elsewhere in the app — no separate "
        "model, no new assumptions — and turns them into a plain-language verdict."
    )

    product_labels = list(PRODUCTS["Product_ID"] + " – " + PRODUCTS["Product_Name"])
    ac1, ac2 = st.columns(2)
    with ac1:
        assess_store = st.selectbox("Store", STORES, key="assess_store")
    with ac2:
        assess_product_label = st.selectbox("Product", product_labels, key="assess_product")
    assess_sku = assess_product_label.split(" – ")[0]

    row = full_table[(full_table["Store"] == assess_store) & (full_table["SKU"] == assess_sku)]
    if row.empty:
        st.warning("No data available for this combination.")
    else:
        r = row.iloc[0]
        verdict = assess_stock_position(r["Current_Stock"], r["Forecast_During_Lead_Time"], r["Safety_Stock"])

        status_style = {
            "Understocked": ("🔴", "#fce8e6"),
            "Below Target": ("🟠", "#fef3e2"),
            "Well Stocked": ("🟢", "#e6f4ea"),
            "Overstocked": ("🟣", "#f3e8fd"),
            "Insufficient Data": ("⚪", "#f0f0f0"),
        }
        icon, bg = status_style.get(verdict["status"], ("⚪", "#f0f0f0"))

        st.markdown(
            f"""<div style="background-color:{bg}; padding:16px 20px; border-radius:10px;">
            <span style="font-size:20px; font-weight:700;">{icon} {verdict['status']}</span><br/>
            <span style="font-size:14.5px;">{verdict['message']}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        if r["Manual_Adjustment"]:
            st.caption(f"Includes +{r['Manual_Adjustment']:.0f} units logged via Stock Intake for {assess_product_label.split(' – ')[1]} at {assess_store}.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Stock", f"{verdict['current_stock']:.0f}")
        m2.metric("Forecast During Lead Time", f"{verdict['forecast_during_lead_time']:.0f}")
        m3.metric("Reorder Point (+20% safety)", f"{verdict['reorder_point']:.0f}")
        m4.metric("Overstock Threshold (2×)", f"{verdict['overstock_threshold']:.0f}" if verdict['overstock_threshold'] else "—")

        if verdict["overstock_threshold"]:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[verdict["current_stock"]], y=["Current Stock"], orientation="h",
                marker_color={"Understocked": "#d62728", "Below Target": "#f77f00", "Well Stocked": "#2ca02c", "Overstocked": "#7b2cbf"}.get(verdict["status"], "#888"),
                showlegend=False,
            ))
            fig.add_vline(x=verdict["forecast_during_lead_time"], line_dash="dash", line_color="#d62728",
                          annotation_text="Bare minimum", annotation_position="top")
            fig.add_vline(x=verdict["reorder_point"], line_dash="dash", line_color="#2ca02c",
                          annotation_text="Reorder point", annotation_position="top")
            fig.add_vline(x=verdict["overstock_threshold"], line_dash="dash", line_color="#7b2cbf",
                          annotation_text="Overstock threshold", annotation_position="top")
            fig.update_layout(height=180, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Units")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Intake ledger (this session)**")
    if not st.session_state.stock_ledger:
        st.caption("No manual stock entries logged yet.")
    else:
        ledger_df = pd.DataFrame(st.session_state.stock_ledger)
        st.dataframe(ledger_df, use_container_width=True, hide_index=True)

        summary = ledger_df.groupby(["Store", "SKU", "Product"])["Quantity"].sum().reset_index().rename(columns={"Quantity": "Total Logged"})
        st.markdown("**Net adjustment by Store × SKU**")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear all logged entries"):
            st.session_state.stock_ledger = []
            st.rerun()

    st.caption(
        "Entries are kept only for this browser session (in memory) — they reset if the app restarts "
        "or the page is closed. In a production system this would write to a real inventory database."
    )

# ===========================================================================
# TAB 5: LEAD-TIME TRACKING — all stores, all SKUs
# ===========================================================================
with tab_leadtime:
    st.subheader("Lead-Time Tracking — All Stores × All SKUs")
    st.caption("Incoming stock shown here is a SIMULATED ESTIMATE (average stock received over the "
               "last Lead-Time-Days days), since the dataset does not record pending purchase orders explicitly.")

    lt_display = full_table[[
        "Store", "Product", "SKU", "Lead_Time_Days", "Forecast_During_Lead_Time",
        "Current_Stock", "Incoming_Stock_Estimate",
    ]].rename(columns={
        "Lead_Time_Days": "Supplier Lead Time (days)", "Forecast_During_Lead_Time": "Forecasted Demand During Lead Time",
        "Current_Stock": "Current Stock", "Incoming_Stock_Estimate": "Incoming Stock (est.)",
    })
    st.dataframe(lt_display, use_container_width=True, hide_index=True)
    st.caption(f"As of {AS_OF_DATE.date()} · Forecast period = {forecast_period} days.")

    fig = px.bar(full_table, x="Store", y=["Current_Stock", "Incoming_Stock_Estimate", "Forecast_During_Lead_Time"],
                 barmode="group", facet_col="Product",
                 title="Current Stock vs Forecasted Demand During Lead Time, by Store & SKU",
                 labels={"value": "Units", "variable": "Metric"})
    st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB: LEAD-TIME ANTICIPATION — predicts actual delivery variability
# ===========================================================================
with tab_leadtime_ai:
    st.subheader("Lead-Time Anticipation — All Stores × All SKUs")
    st.write(
        "The historical dataset records a single **fixed** lead time per SKU, with no day-to-day "
        "variability to learn from — so a real supplier never delivers in exactly the same number "
        "of days every time. This tab **anticipates** the next delivery using variability actually "
        "observed in the Live Simulation, rather than just repeating that fixed textbook number."
    )

    sim_state = st.session_state.get("sim_state")
    if sim_state is None or sim_state["lead_time_log"].empty:
        st.info(
            "No delivery history yet — open the **🔴 Live Simulation** tab and advance a few days "
            "(or turn on Auto-play). As orders are placed and arrive, this tab will start anticipating "
            "each Store × SKU's real delivery variability from what actually happened.",
            icon="🚚",
        )
        st.markdown("**Planned lead time (fixed, from the dataset) — shown until simulation data exists:**")
        st.dataframe(
            full_table[["Store", "Product", "SKU", "Lead_Time_Days"]].rename(columns={"Lead_Time_Days": "Planned Lead Time (days)"}),
            use_container_width=True, hide_index=True,
        )
    else:
        lt_log = sim_state["lead_time_log"]
        rows = []
        for _, r in full_table.iterrows():
            res = simulation.anticipate_lead_time(lt_log, r["Store"], r["SKU"], r["Lead_Time_Days"])
            rows.append({
                "Store": r["Store"], "Product": r["Product"], "SKU": r["SKU"],
                "Planned Lead Time (days)": r["Lead_Time_Days"],
                "Deliveries Observed": res["n_observations"],
                "Anticipated Lead Time (days)": res["anticipated_days"],
                "Range (± 1 std dev)": f"{res['low']}–{res['high']}" if res["n_observations"] > 0 else "—",
            })
        anticipation_df = pd.DataFrame(rows)
        st.dataframe(anticipation_df, use_container_width=True, hide_index=True)
        st.caption(f"Based on {len(lt_log)} order(s) observed across {sim_state['days_simulated']} simulated day(s) so far.")

        st.divider()
        st.markdown("**Planned vs actual delivery time — every logged order**")
        plot_df = lt_log.copy()
        plot_df["Combo"] = plot_df["Store"] + " · " + plot_df["Product_Name"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df["Order_Date"], y=plot_df["Planned_Lead_Time"], mode="markers",
                                  name="Planned", marker=dict(color="#9467bd", symbol="line-ew", size=10)))
        fig.add_trace(go.Scatter(x=plot_df["Order_Date"], y=plot_df["Actual_Lead_Time"], mode="markers",
                                  name="Actual", marker=dict(color="#2ca02c", size=7)))
        fig.update_layout(title="Planned vs Actual Lead Time — every order placed in the simulation",
                           xaxis_title="Order Date", yaxis_title="Lead Time (days)",
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "SIMULATED: this variability comes from an explicit, disclosed assumption in the Live "
            "Simulation (± 30% spread around each SKU's fixed lead time), used to give this "
            "anticipation feature genuine variability to learn from. It is not measured from real "
            "supplier performance data, which this dataset does not contain."
        )

# ===========================================================================
# TAB 6: EXPIRY TRACKING — all stores, all SKUs
# ===========================================================================
with tab_expiry:
    st.subheader("Expiry Tracking — All Stores × All SKUs")
    st.warning(
        "**Simulated/prototype values.** This dataset is not connected to Colonial Fresh's live "
        "operational systems. Days Remaining reflects the shelf life of the most recently received "
        "stock lot as of the dataset's latest date — a simplification appropriate for this Sprint 2 design.",
        icon="⚠️",
    )

    exp_display = full_table.copy()
    exp_display["Expiry Risk"] = exp_display["Days_To_Expiry"].apply(
        lambda d: "Expiry Risk" if d <= 2 else ("Watch" if d <= 4 else "OK")
    )
    exp_display = exp_display[[
        "Store", "Product", "SKU", "Shelf_Life_Days", "Expiry_Date", "Days_To_Expiry", "Expiry Risk",
    ]].rename(columns={"Shelf_Life_Days": "Shelf Life (days)", "Expiry_Date": "Expiry Date", "Days_To_Expiry": "Days Remaining"})

    def _expiry_style(val):
        colors = {"Expiry Risk": "background-color:#f3e8fd", "Watch": "background-color:#fef3e2", "OK": "background-color:#e6f4ea"}
        return colors.get(val, "")

    st.dataframe(exp_display.style.map(_expiry_style, subset=["Expiry Risk"]), use_container_width=True, hide_index=True)
    st.caption(f"As of {AS_OF_DATE.date()}")

# ===========================================================================
# TAB 7: DEMAND-BASED REORDER RECOMMENDATION — all stores, all SKUs
# ===========================================================================
with tab_reorder:
    st.subheader("Demand-Based Reorder Recommendation — All Stores × All SKUs")

    st.markdown("**Formula:** `Recommended Order = Forecasted Demand During Lead Time + Safety Stock − Current Stock − Incoming Stock`")
    st.markdown("**Safety Stock = 20% of Forecasted Demand During Lead Time**")
    st.caption(
        "The system recommends an order whenever available inventory (plus what's already inbound) "
        "is below the expected demand requirement during the replenishment period."
    )

    total_recommended = full_table["Recommended_Order"].sum()
    combos_needing_order = (full_table["Recommended_Order"] > 0).sum()
    m1, m2 = st.columns(2)
    m1.metric("Total Recommended Reorder (network)", f"{total_recommended:,.0f} units")
    m2.metric("Store × SKU combos needing an order", f"{combos_needing_order} / {len(full_table)}")

    reorder_display = full_table[[
        "Store", "Product", "SKU", "Forecast_During_Lead_Time", "Safety_Stock",
        "Current_Stock", "Incoming_Stock_Estimate", "Raw_Calculation", "Recommended_Order", "Risk",
    ]].rename(columns={
        "Forecast_During_Lead_Time": "Forecast During Lead Time", "Safety_Stock": "Safety Stock (20%)",
        "Current_Stock": "Current Stock", "Incoming_Stock_Estimate": "Incoming Stock (est.)",
        "Raw_Calculation": "Raw Calculation", "Recommended_Order": "Recommended Order",
    })
    st.dataframe(reorder_display.style.map(_risk_style, subset=["Risk"]), use_container_width=True, hide_index=True)

    if combos_needing_order > 0:
        st.success(
            f"{combos_needing_order} store × SKU combination(s) currently need a reorder — "
            f"see the highlighted rows above for the full step-by-step calculation."
        )
    else:
        st.success("No reorders are currently recommended anywhere in the network.")

# ===========================================================================
# TAB 8: RISK DASHBOARD
# ===========================================================================
with tab_risk:
    st.subheader("Risk Dashboard")
    st.caption(f"Dynamically calculated for all 6 stores × 3 SKUs · as of {AS_OF_DATE.date()} · "
               f"forecast period = {forecast_period} days.")

    display_cols = ["Store", "SKU", "Product", "Movement", "Current_Stock",
                     "Forecast_Period_Demand", "Lead_Time_Days", "Days_To_Expiry", "Risk", "Recommended_Order"]
    display_df = risk_table[display_cols].rename(columns={
        "Current_Stock": "Current Stock", "Forecast_Period_Demand": "Forecast",
        "Lead_Time_Days": "Lead Time (d)", "Days_To_Expiry": "Expiry (days left)",
        "Recommended_Order": "Recommended Order",
    })

    st.dataframe(
        display_df.style.map(_risk_style, subset=["Risk"]),
        use_container_width=True, hide_index=True,
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        wastage_by_sku = filtered_df.groupby("Product_Name")["Wastage"].sum().reset_index()
        fig = px.bar(wastage_by_sku, x="Product_Name", y="Wastage", title="Wastage by SKU (full history, current scope)",
                     labels={"Product_Name": "Product"}, color="Product_Name")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        stockout_counts = filtered_df[filtered_df["Stockout"] == "Yes"].groupby("Product_Name").size().reset_index(name="Stockout Days")
        fig = px.bar(stockout_counts, x="Product_Name", y="Stockout Days", title="Stockout Occurrences by SKU (full history, current scope)",
                     labels={"Product_Name": "Product"}, color="Product_Name")
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB: LIVE SIMULATION (full network — all 6 stores x 3 SKUs)
# ===========================================================================
with tab_live:
    st.subheader("Live Simulation — Full Store Network")
    st.info(
        "This simulates ongoing daily grocery operations **forward** from the end of the "
        "real historical dataset (i.e. from " + str(AS_OF_DATE.date()) + " onward). Each simulated "
        "day's demand is synthetically generated from patterns learned from the real 2024-2025 "
        "history for each Store+SKU (recent average daily sales, day-of-week seasonality, "
        "promotion effect size) plus random day-to-day variation — **it is not real sales data**. "
        "The same 7-day moving-average forecast, lead-time-aware reorder formula and risk "
        "classification used elsewhere in this app run automatically every simulated day, so you "
        "can watch demand-based reordering actually trigger and restocks actually arrive.",
        icon="🔴",
    )

    if "sim_state" not in st.session_state:
        st.session_state.sim_state = simulation.init_simulation_state(df, STORES, PRODUCTS, AS_OF_DATE, seed=None)
    if "sim_autoplay" not in st.session_state:
        st.session_state.sim_autoplay = False
    if "sim_speed" not in st.session_state:
        st.session_state.sim_speed = 1.5

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1.4, 1])
    with ctrl1:
        next_day_clicked = st.button("▶ Next Day", use_container_width=True)
    with ctrl2:
        reset_clicked = st.button("🔁 Reset Simulation", use_container_width=True)
    with ctrl3:
        st.session_state.sim_speed = st.slider("Auto-play speed (seconds/day)", 0.3, 4.0, st.session_state.sim_speed, 0.1)
    with ctrl4:
        st.session_state.sim_autoplay = st.toggle("Auto-play", value=st.session_state.sim_autoplay)

    if reset_clicked:
        st.session_state.sim_state = simulation.init_simulation_state(df, STORES, PRODUCTS, AS_OF_DATE, seed=None)
        st.session_state.sim_autoplay = False
        st.rerun()

    if next_day_clicked:
        st.session_state.sim_state = simulation.simulate_next_day(st.session_state.sim_state)

    sim_state = st.session_state.sim_state
    sim_log = sim_state["log"]

    st.markdown(f"**Simulated date:** {sim_state['sim_date'].date()}  ·  "
                f"**Days simulated:** {sim_state['days_simulated']}")

    if sim_log.empty:
        st.warning("No simulated days yet — click **Next Day** or turn on **Auto-play** to begin.")
    else:
        latest_date = sim_log["Date"].max()
        today_log = sim_log[sim_log["Date"] == latest_date]

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Network Stock (units)", f"{today_log['Closing_Stock'].sum():,.0f}")
        k2.metric("Units Sold Today", f"{today_log['Units_Sold'].sum():,.0f}")
        k3.metric("Wastage Today", f"{today_log['Wastage'].sum():,.0f}")
        k4.metric("Orders Placed Today", f"{(today_log['Recommended_Order'] > 0).sum()} combos")
        k5.metric("Units Reordered Today", f"{today_log['Recommended_Order'].sum():,.0f}")

        st.divider()
        st.markdown("**Today's snapshot — all stores × SKUs**")

        show_cols = ["Store", "Product_Name", "Opening_Stock", "Incoming_Stock", "Units_Sold",
                     "Closing_Stock", "Wastage", "Stockout", "Recommended_Order", "Risk"]
        st.dataframe(
            today_log[show_cols].rename(columns={"Product_Name": "Product"}).style.map(_risk_style, subset=["Risk"]),
            use_container_width=True, hide_index=True,
        )

        st.caption(
            "**Reading this table:** 'Risk' is a leading, forward-looking flag based on whether "
            "on-hand + already-ordered stock covers a full lead-time cycle under the 20% safety-stock "
            "policy. 'Stockout' shows whether that SPECIFIC day's actual demand was fully met. With a "
            "lean 20% safety margin and short 1-2 day lead times, 'Stockout Risk' can flag often as an "
            "early warning even while actual demand keeps getting met most days — the two metrics below "
            "make that gap visible, which is a genuine discussion point for Sprint 3 (e.g. would a higher "
            "safety-stock % reduce false-positive risk flags at the cost of holding more stock?)."
        )

        c1, c2 = st.columns(2)
        with c1:
            actual_stockout_rate = (sim_log["Stockout"] == "Yes").mean()
            st.metric("Actual demand-unmet rate (measured, all days simulated)", f"{actual_stockout_rate:.1%}")
        with c2:
            risk_flag_rate = (sim_log["Risk"] != "Normal").mean()
            st.metric("Risk-flag rate (leading indicator, all days simulated)", f"{risk_flag_rate:.1%}")

        st.divider()
        st.markdown("**Network trends across the simulation**")
        network_daily = sim_log.groupby("Date").agg(
            Total_Units_Sold=("Units_Sold", "sum"),
            Total_Forecast=("Avg_Daily_Forecast", "sum"),
            Total_Closing_Stock=("Closing_Stock", "sum"),
            Total_Wastage=("Wastage", "sum"),
        ).reset_index()

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=network_daily["Date"], y=network_daily["Total_Units_Sold"],
                                   name="Actual Units Sold (network)", line=dict(color="#1f77b4")))
        fig1.add_trace(go.Scatter(x=network_daily["Date"], y=network_daily["Total_Forecast"],
                                   name="Forecast (sum of daily rates, network)", line=dict(color="#2ca02c", dash="dash")))
        fig1.update_layout(title="Network-Wide Actual Demand vs Forecast (simulated days)",
                           xaxis_title="Date", yaxis_title="Units")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=network_daily["Date"], y=network_daily["Total_Closing_Stock"],
                                   name="Total Network Closing Stock", line=dict(color="#1f77b4")))
        fig2.update_layout(title="Total Network Stock On Hand Over Time", xaxis_title="Date", yaxis_title="Units")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Stock level by SKU, all stores combined**")
        stock_by_sku = sim_log.groupby(["Date", "Product_Name"])["Closing_Stock"].sum().reset_index()
        fig3 = px.line(stock_by_sku, x="Date", y="Closing_Stock", color="Product_Name",
                        title="Closing Stock Over Time by SKU (summed across all 6 stores)",
                        labels={"Closing_Stock": "Closing Stock", "Product_Name": "Product"})
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("**Reorders placed by store, all SKUs combined**")
        orders_by_store = sim_log.groupby(["Date", "Store"])["Recommended_Order"].sum().reset_index()
        orders_by_store = orders_by_store[orders_by_store["Recommended_Order"] > 0]
        if orders_by_store.empty:
            st.caption("No reorders placed yet.")
        else:
            fig4 = px.bar(orders_by_store, x="Date", y="Recommended_Order", color="Store",
                          title="Units Reordered per Day, by Store", labels={"Recommended_Order": "Units Reordered"})
            st.plotly_chart(fig4, use_container_width=True)

    if st.session_state.sim_autoplay:
        time.sleep(st.session_state.sim_speed)
        st.session_state.sim_state = simulation.simulate_next_day(st.session_state.sim_state)
        st.rerun()

# ===========================================================================
# TAB: SOLUTION ARCHITECTURE (Sprint 2 design explanation)
# ===========================================================================
with tab_architecture:
    st.subheader("Solution Architecture — Sprint 2 Design")
    st.markdown(
        """
This Sprint 2 prototype demonstrates the proposed end-to-end design. Sprint 3 will formally
develop and demonstrate the working prototype in full; this sprint focuses on **conceptualisation,
design and technology integration**.

**Data & processing pipeline:**

```
Excel Dataset (Colonial Fresh Mock Data)
        ↓
Data Cleaning & Preparation   (pandas: date conversion, type correction, missing-value handling, validation)
        ↓
Product Movement Analysis     (average daily Units_Sold per SKU → Fast/Moderate/Slow classification)
        ↓
Demand Forecasting            (7-day moving-average baseline; optional scikit-learn Linear Regression trend for comparison)
        ↓
Inventory Analysis            (Opening/Incoming/Sold/Closing/Wastage/Stockout monitoring)
        ↓
Lead-Time & Expiry Tracking   (supplier lead time, forecasted demand during lead time, shelf-life/expiry)
        ↓
Risk Detection                 (Normal / Stockout Risk / Overstock Risk / Expiry Risk classification)
        ↓
Demand-Based Reorder Calculation  (Forecast during lead time + 20% safety stock − current − incoming)
        ↓
Streamlit Dashboard             (Overview, Movement, Forecasting, Inventory, Lead-Time, Expiry, Reorder, Risk views)
```

**Technology stack:** Python, Pandas (data handling), Scikit-learn (optional comparison forecasting
model), Streamlit (interactive dashboard UI), Plotly (interactive charts).

**Why this design fits Sprint 2:** it is simple and fully explainable (baseline moving-average
forecast + transparent reorder formula), directly implements all six points raised in lecturer
feedback (three-SKU baseline, demand forecasting, stock dashboard, lead-time tracking, expiry
tracking, demand-based reorder), and is modular — each pipeline stage is a separate, testable
Python module (`data_utils.py`, `forecasting.py`, `inventory_analysis.py`), so it can be extended
in Sprint 3 without rebuilding the design.

**Potential benefits (proposed, not yet measured):** more consistent reorder decisions, earlier
visibility of stockout/overstock/expiry risk, and a single dashboard combining data that is
currently split across separate manual processes. These are proposed benefits for future
validation in Sprint 3 — no business-improvement results are claimed from this Sprint 2 prototype.

**Future development (beyond Sprint 2):** live POS/ERP integration in place of the Excel file,
batch-level (FEFO) expiry tracking, back-tested forecast accuracy comparison between the baseline
and ML models, and automated purchase-order generation from the reorder recommendation.
        """
    )

st.divider()
st.caption(
    "Colonial Fresh AI-Based Demand Forecasting and Inventory Management System — "
    "Sprint 2 academic prototype. Simulated data only. Not a live business system."
)
