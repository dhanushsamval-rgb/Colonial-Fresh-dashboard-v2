# Colonial Fresh — AI-Based Demand Forecasting & Inventory Management System

**MGT5ELS Assessment 3 — Sprint 2 (Conceptualisation & Design) prototype**

A functional demonstration of the proposed design for a fresh-food retail demand-forecasting
and inventory-decision-support dashboard, built on a fully **simulated** 2024–2025 dataset for
three baseline SKUs (Bananas, Apples, Strawberries) across six Colonial Fresh stores.

> **Academic integrity note:** the dataset (`data/Colonial_Fresh_Mock_Data_2024_2025_3_SKUs.xlsx`)
> is entirely simulated for this assessment. It is not real Colonial Fresh data and this
> application is not connected to any live business system.

## What this Sprint 2 prototype covers

**This dashboard always shows all 6 stores × all 3 SKUs together — nothing is filtered or
hidden anywhere.** The only control in the sidebar is the forecast horizon (7/14/30 days),
since that's a genuine parameter choice rather than something that hides data. This was a
deliberate design choice to make the app easy to present end-to-end without pausing to
change dropdowns.

1. Overview Dashboard (network-wide KPIs)
2. Product Movement Analysis (data-driven fast/moderate/slow classification)
3. Demand Forecasting (7-day moving-average baseline, aggregated across all stores per SKU; optional scikit-learn linear-trend comparison)
4. Inventory Monitoring (Opening/Incoming/Sold/Closing/Wastage/Stockout — every store × SKU in one table)
5. Lead-Time Tracking (forecast during lead time — every store × SKU)
6. Expiry Tracking (shelf life / expiry date / days remaining — every store × SKU, labelled as simulated)
7. Demand-Based Reorder Recommendation (full formula breakdown — every store × SKU)
8. Risk Dashboard (Normal / Stockout Risk / Overstock Risk / Expiry Risk, all 18 combos)
9. **Live Simulation** (full 6-store × 3-SKU network, running forward day-by-day — see below)
10. **Lead-Time Anticipation** — predicts the next delivery's actual lead time (with a range), learned from real variability observed as the Live Simulation runs, rather than repeating the dataset's fixed textbook value
11. **Stock Intake** — manually log stock receipts; every entry genuinely shifts Current Stock and flows through to every other tab's calculations
12. **Machine Learning Forecast** — a real scikit-learn Random Forest model, chronologically backtested against the baseline with honest accuracy metrics (MAE/RMSE/MAPE), not just a side comparison
13. Solution Architecture (the Sprint 2 design write-up, embedded in the app)

## Lead-Time Anticipation, Stock Intake & Machine Learning Forecast

Three deeper additions beyond the original Sprint 2 brief:

- **Lead-Time Anticipation** (`simulation.py`, `anticipate_lead_time()`): the dataset only records
  a single fixed lead time per SKU with no real variability to learn from. To make "anticipation"
  meaningful rather than just repeating a constant, the Live Simulation was extended with an
  explicitly disclosed ±30% variability assumption around each SKU's base lead time. As the
  simulation places and receives orders, this tab tracks the actual vs planned lead time for every
  order and reports a genuine data-driven anticipated delivery window (mean ± 1 std dev). Before
  any simulation has run, it honestly falls back to the fixed dataset value and says so.
- **Stock Intake** (`app.py`, Stock Intake tab): a form to manually log stock receipts during a
  session. Entries are kept in `st.session_state` and passed as a `stock_adjustments` dict into
  both `build_full_snapshot_table()` and `build_risk_table()` (`inventory_analysis.py`), so a
  logged receipt genuinely changes Current Stock and therefore every downstream lead-time, reorder
  and risk calculation across the whole app — not just a note on one tab. Entries reset when the
  app restarts (session-only, clearly disclosed in-app). Right below the form, an **AI Stock
  Assessment** panel (`inventory_analysis.assess_stock_position()`) reads the same forecast and
  reorder-point numbers already used everywhere else and gives a plain-language verdict —
  Understocked / Below Target / Well Stocked / Overstocked — against the same 20%-safety-stock and
  2x-overstock thresholds the rest of the app uses, so the verdict is never a separate opinion.
- **Machine Learning Forecast** (`ml_forecast.py`): a real `RandomForestRegressor` trained per SKU
  on lag/rolling/calendar features (previous day, previous week, 7- and 14-day rolling averages,
  day-of-week, promotion rate), evaluated with a genuine **chronological** train/test split (last
  60 days held out, never seen during training) — never a random split, to avoid leakage.
  MAE/RMSE/MAPE are reported for both the Random Forest and the baseline on the exact same test
  window, so any claim of one beating the other is backed by a real backtest. `Simulated_True_Demand`
  is still never used as a feature or target. Forward forecasts beyond the dataset's last date use
  recursive multi-step prediction (each predicted day feeds the next day's lag features).

## Live Simulation tab

This is the most advanced piece of the prototype: a day-by-day simulation that runs the whole
demand → forecast → reorder → restock loop live, across all 6 stores and all 3 SKUs, starting
the day after the real historical data ends.

- Each simulated day's demand is generated from patterns learned from the real history (recent
  average sales, day-of-week seasonality, promotion effect size) plus random daily variation —
  it is **not** real or predicted future sales, purely a plausible synthetic continuation used to
  demonstrate the decision logic in motion.
- The exact same forecasting (`forecasting.py`), reorder formula and risk classification
  (`inventory_analysis.py`) used in the historical tabs are reused here, so the two parts of the
  app stay consistent.
- When the system recommends a reorder, it is automatically "placed" and arrives after that SKU's
  lead time — so you can watch stock deplete, a reorder trigger, and the restock arrive on
  schedule, all live.
- Controls: **Next Day** (manual step), **Reset Simulation**, and an **Auto-play** toggle with a
  speed slider.
- Two KPIs are shown side by side on purpose: the **Risk-flag rate** (a leading, forward-looking
  warning based on whether on-hand + on-order stock covers a full lead-time cycle under the 20%
  safety-stock policy) and the **actual demand-unmet rate** (a lagging, measured outcome — whether
  that day's real demand was actually fulfilled). In testing, the risk flag fires on ~99% of
  combo-days while actual unmet demand sits around ~13% — a genuine finding worth discussing with
  your lecturer: with only a 20% safety margin and 1–2 day lead times, this reorder policy keeps
  inventory lean enough that the leading-indicator flag fires far more often than real shortages
  occur. A natural Sprint 3 question: would a higher safety-stock percentage reduce false-positive
  risk flags, at the cost of holding more stock?

## How to run it

1. Install Python 3.10+.
2. From this folder, install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Launch the dashboard:
   ```
   streamlit run app.py
   ```
4. Your browser will open automatically at `http://localhost:8501`.

The Excel dataset is already included in `data/`, so no extra setup is needed. If you replace
it with an updated file, keep the same filename and sheet name (`Mock_Data`).

## Project structure

```
colonial_fresh_dashboard/
├── app.py                  # Streamlit UI — all dashboard tabs, incl. ML Forecast, Stock Intake, Lead-Time Anticipation
├── data_utils.py            # Loading, validation, cleaning
├── forecasting.py           # Baseline moving-average model + optional ML comparison model
├── inventory_analysis.py    # Movement analysis, lead-time, expiry, reorder, risk logic (with stock-adjustment support)
├── simulation.py            # Live day-by-day simulation engine (full store network) + lead-time variability tracking
├── ml_forecast.py           # Backtested Random Forest demand forecast
├── requirements.txt
├── README.md
└── data/
    └── Colonial_Fresh_Mock_Data_2024_2025_3_SKUs.xlsx
```

## Key design decisions (for your lecturer discussion)

- **Forecasting:** the primary model is a simple, explainable 7-day moving average (as required
  by the brief). An optional scikit-learn `LinearRegression` trend model is shown side-by-side
  purely for comparison — it is **not** claimed to be more accurate, since no back-testing has
  been done in this sprint. `Simulated_True_Demand` is never used as a model input, to avoid
  data leakage; it is reserved for future evaluation.
- **Reorder formula:** `Recommended Order = Forecasted Demand During Lead Time + Safety Stock
  − Current Stock − Incoming Stock`, with `Safety Stock = 20% of Forecasted Demand During Lead
  Time`, floored at 0. Shown step-by-step in the app.
- **"Real-time" framing:** the dashboard treats the most recent date in the dataset as "today"
  to demonstrate real-time-style behaviour. Live POS/ERP integration is proposed as Sprint 3+
  future work, not claimed as already built.
- **Incoming stock (pipeline) estimate:** the dataset doesn't track pending purchase orders, so
  incoming stock for lead-time/reorder purposes is estimated from recent historical receipts.
  This is disclosed in the app.

## Testing performed (Sprint 2 scope)

All three baseline SKUs were tested against at least one store each, confirming that:
forecasts differ by product, inventory values update per store/SKU, lead time is incorporated
into the reorder calculation, expiry information displays correctly, risk status is calculated
per combination, and reorder quantities are calculated dynamically (including cases where the
recommended order is greater than zero, e.g. Apples at Doncaster and The Pines, and Strawberries
at The Pines, under Stockout Risk).
