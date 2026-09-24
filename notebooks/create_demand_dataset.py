from pathlib import Path
import pandas as pd

# Project root
BASE_DIR = Path(__file__).resolve().parents[1]

# Input
input_file = (
    BASE_DIR /
    "data" /
    "processed" /
    "online_retail_cleaned.csv"
)

# Output
output_file = (
    BASE_DIR /
    "data" /
    "processed" /
    "product_monthly_demand.csv"
)

print("Loading cleaned dataset...")

df = pd.read_csv(input_file)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# ------------------------------------------------
# Create month
# ------------------------------------------------

df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

# ------------------------------------------------
# Aggregate product + month
# ------------------------------------------------

monthly = (
    df.groupby(
        ["StockCode", "Description", "Month"]
    )
    .agg(
        Quantity=("Quantity", "sum"),
        Revenue=("Revenue", "sum"),
        Transactions=("Invoice", "count"),
        Customers=("Customer ID", "nunique"),
        Avg_Price=("Price", "mean"),
        Min_Price=("Price", "min"),
        Max_Price=("Price", "max")
    )
    .reset_index()
)

# ------------------------------------------------
# Calculate revenue per unit
# ------------------------------------------------

monthly["Revenue_Per_Unit"] = (
    monthly["Revenue"] /
    monthly["Quantity"]
)

# ------------------------------------------------
# Sort
# ------------------------------------------------

monthly = monthly.sort_values(
    ["StockCode", "Month"]
)

# ------------------------------------------------
# Save
# ------------------------------------------------

monthly.to_csv(
    output_file,
    index=False
)

print("\n========== MONTHLY DEMAND DATASET ==========")

print(f"Rows: {len(monthly):,}")
print(f"Columns: {len(monthly.columns)}")

print("\nColumns:")
print(monthly.columns.tolist())

print("\nFirst 10 rows:")
print(monthly.head(10).to_string(index=False))

print("\nSaved to:")
print(output_file)