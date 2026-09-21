"""
data_utils.py
--------------
Data loading, cleaning and validation for the Colonial Fresh prototype.

IMPORTANT (academic integrity note):
The Excel file used here (Colonial_Fresh_Mock_Data_2024_2025_3_SKUs.xlsx) is a
FULLY SIMULATED dataset built for this university assessment. It is NOT real
Colonial Fresh operational data and must never be presented as such.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).parent / "data" / "Colonial_Fresh_Mock_Data_2024_2025_3_SKUs.xlsx"

REQUIRED_COLUMNS = [
    "Date", "Store", "Product_ID", "Product_Name", "Movement_Class",
    "Opening_Stock", "Incoming_Stock", "Simulated_True_Demand", "Units_Sold",
    "Closing_Stock", "Wastage", "Promotion", "Selling_Price_AUD", "Stockout",
    "Lead_Time_Days", "Shelf_Life_Days", "Expiry_Date",
]

# Baseline SKUs required by the assessment brief.
BASELINE_SKUS = ["SKU001", "SKU002", "SKU003"]


@st.cache_data(show_spinner=False)
def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw Excel dataset (Mock_Data sheet)."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the Colonial Fresh mock data "
            f"Excel file inside the /data folder."
        )
    df = pd.read_excel(path, sheet_name="Mock_Data")
    return df


def validate_columns(df: pd.DataFrame) -> list:
    """Return a list of missing required columns (empty list = OK)."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and type-correct the raw dataset. Raises a clear error rather than
    silently inventing data if something unexpected is found.
    """
    df = df.copy()

    # --- Date conversion -------------------------------------------------
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Expiry_Date"] = pd.to_datetime(df["Expiry_Date"], errors="coerce")

    bad_dates = df["Date"].isna().sum()
    if bad_dates > 0:
        # Do not silently invent dates - drop and flag instead.
        df = df.dropna(subset=["Date"])

    # --- Numeric type correction -----------------------------------------
    numeric_cols = [
        "Opening_Stock", "Incoming_Stock", "Simulated_True_Demand",
        "Units_Sold", "Closing_Stock", "Wastage", "Selling_Price_AUD",
        "Lead_Time_Days", "Shelf_Life_Days",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Missing value handling -------------------------------------------
    # We do NOT invent sales/stock figures. Rows with missing critical
    # numeric fields are excluded and counted, so the impact is visible
    # rather than hidden.
    critical = ["Opening_Stock", "Incoming_Stock", "Units_Sold", "Closing_Stock", "Wastage"]
    before = len(df)
    df = df.dropna(subset=critical)
    dropped = before - len(df)

    # --- Categorical/text normalisation -----------------------------------
    df["Store"] = df["Store"].astype(str).str.strip()
    df["Product_ID"] = df["Product_ID"].astype(str).str.strip()
    df["Product_Name"] = df["Product_Name"].astype(str).str.strip()
    df["Movement_Class"] = df["Movement_Class"].astype(str).str.strip()
    df["Stockout"] = df["Stockout"].astype(str).str.strip()
    df["Promotion"] = df["Promotion"].astype(str).str.strip()

    # --- Sort for time-series operations (moving averages etc.) ----------
    df = df.sort_values(["Store", "Product_ID", "Date"]).reset_index(drop=True)

    df.attrs["rows_dropped_missing"] = dropped
    df.attrs["rows_dropped_bad_dates"] = bad_dates
    return df


def get_stores(df: pd.DataFrame) -> list:
    return sorted(df["Store"].unique().tolist())


def get_products(df: pd.DataFrame) -> pd.DataFrame:
    """Return unique Product_ID / Product_Name / Movement_Class combinations."""
    return (
        df[["Product_ID", "Product_Name", "Movement_Class"]]
        .drop_duplicates()
        .sort_values("Product_ID")
        .reset_index(drop=True)
    )


def get_as_of_date(df: pd.DataFrame) -> pd.Timestamp:
    """
    The most recent date in the dataset. In this prototype we treat this
    date as 'today' for inventory / lead-time / expiry snapshots, since the
    dataset is historical (2024-2025) rather than a live feed.
    """
    return df["Date"].max()


def filter_data(df: pd.DataFrame, stores=None, products=None) -> pd.DataFrame:
    """Filter by a list of stores and/or product IDs. None/empty = all."""
    out = df
    if stores:
        out = out[out["Store"].isin(stores)]
    if products:
        out = out[out["Product_ID"].isin(products)]
    return out


def load_and_prepare() -> tuple:
    """
    Convenience wrapper: load raw data, validate schema, clean it.
    Returns (df_clean, validation_messages).
    """
    messages = []
    raw = load_raw_data()
    missing = validate_columns(raw)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Cannot proceed without inventing data."
        )
    clean = clean_data(raw)
    if clean.attrs.get("rows_dropped_missing", 0) > 0:
        messages.append(
            f"{clean.attrs['rows_dropped_missing']} row(s) were removed due to "
            f"missing critical values (no values were invented)."
        )
    if clean.attrs.get("rows_dropped_bad_dates", 0) > 0:
        messages.append(
            f"{clean.attrs['rows_dropped_bad_dates']} row(s) were removed due to "
            f"unparseable dates."
        )
    missing_skus = [s for s in BASELINE_SKUS if s not in clean["Product_ID"].unique()]
    if missing_skus:
        messages.append(
            f"Warning: expected baseline SKUs not found in dataset: {missing_skus}"
        )
    return clean, messages
