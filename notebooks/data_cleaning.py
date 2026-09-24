from pathlib import Path
import pandas as pd

# Project root
BASE_DIR = Path(__file__).resolve().parents[1]

# Input file
input_file = BASE_DIR / "data" / "raw" / "online_retail_II.xlsx"

# Output file
output_file = BASE_DIR / "data" / "processed" / "online_retail_cleaned.csv"

print("Loading dataset...")

# Read both sheets
df_2009 = pd.read_excel(
    input_file,
    sheet_name="Year 2009-2010"
)

df_2010 = pd.read_excel(
    input_file,
    sheet_name="Year 2010-2011"
)

# Combine both years
df = pd.concat(
    [df_2009, df_2010],
    ignore_index=True
)

print(f"Original rows: {len(df):,}")

# ------------------------------------------------
# 1. Remove cancelled transactions
# ------------------------------------------------

cancelled = df["Invoice"].astype(str).str.startswith("C")

print(f"Cancelled transactions: {cancelled.sum():,}")

df = df[~cancelled].copy()

# ------------------------------------------------
# 2. Remove rows with missing Customer ID
# ------------------------------------------------

missing_customer = df["Customer ID"].isna()

print(f"Missing Customer ID rows: {missing_customer.sum():,}")

df = df[~missing_customer].copy()

# ------------------------------------------------
# 3. Remove invalid quantities
# ------------------------------------------------

invalid_quantity = df["Quantity"] <= 0

print(f"Invalid quantity rows: {invalid_quantity.sum():,}")

df = df[~invalid_quantity].copy()

# ------------------------------------------------
# 4. Remove invalid prices
# ------------------------------------------------

invalid_price = df["Price"] <= 0

print(f"Invalid price rows: {invalid_price.sum():,}")

df = df[~invalid_price].copy()

# ------------------------------------------------
# 5. Create Revenue column
# ------------------------------------------------

df["Revenue"] = df["Quantity"] * df["Price"]

# ------------------------------------------------
# 6. Convert Customer ID to integer
# ------------------------------------------------

df["Customer ID"] = df["Customer ID"].astype(int)

# ------------------------------------------------
# 7. Sort by Invoice Date
# ------------------------------------------------

df = df.sort_values("InvoiceDate")

# ------------------------------------------------
# 8. Save cleaned dataset
# ------------------------------------------------

df.to_csv(
    output_file,
    index=False
)

print("\nCleaning completed successfully!")

print(f"Final rows: {len(df):,}")
print(f"Final columns: {len(df.columns)}")

print("\nColumns after cleaning:")
print(df.columns.tolist())

print("\nCleaned dataset saved to:")
print(output_file)