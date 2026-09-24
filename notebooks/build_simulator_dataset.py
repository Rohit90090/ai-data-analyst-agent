from pathlib import Path
import pandas as pd


# ==============================
# 1. PROJECT PATH
# ==============================

BASE_DIR = Path(__file__).resolve().parents[1]

input_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "product_monthly_demand.csv"
)


# ==============================
# 2. LOAD DATA
# ==============================

df = pd.read_csv(input_file)

df["Month"] = pd.to_datetime(df["Month"])

print("Original rows:", len(df))


# ==============================
# 3. CLEAN TEXT
# ==============================

df["StockCode"] = (
    df["StockCode"]
    .astype(str)
    .str.strip()
)

df["Description"] = (
    df["Description"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ==============================
# 4. REMOVE NON-PRODUCT CODES
# ==============================

excluded_codes = {
    "POST",
    "M",
    "D",
    "C2",
    "S",
    "DOT",
    "CRUK",
    "AMAZONFEE",
    "BANK CHARGES",
    "SAMPLES"
}

df = df[
    ~df["StockCode"]
    .str.upper()
    .isin(excluded_codes)
].copy()


# ==============================
# 5. REMOVE NON-PRODUCT NAMES
# ==============================

non_product_words = [
    "POSTAGE",
    "CARRIAGE",
    "MANUAL",
    "BANK CHARGE",
    "AMAZON FEE",
    "SAMPLES",
    "DISCOUNT",
    "ADJUSTMENT",
    "DOTCOM POSTAGE"
]

pattern = "|".join(non_product_words)

df = df[
    ~df["Description"]
    .str.upper()
    .str.contains(
        pattern,
        regex=True,
        na=False
    )
].copy()


# ==============================
# 6. REMOVE INVALID DATA
# ==============================

df = df[
    (df["Quantity"] > 0) &
    (df["Revenue"] > 0) &
    (df["Avg_Price"] > 0) &
    (df["Transactions"] > 0) &
    (df["Customers"] > 0)
].copy()


# ==============================
# 7. PRODUCT SUMMARY
# ==============================

product_summary = (
    df.groupby(
        ["StockCode", "Description"]
    )
    .agg(
        Months=("Month", "nunique"),
        Total_Quantity=("Quantity", "sum"),
        Total_Revenue=("Revenue", "sum"),
        Total_Transactions=("Transactions", "sum"),
        Avg_Price=("Avg_Price", "mean"),
        Min_Price=("Min_Price", "min"),
        Max_Price=("Max_Price", "max")
    )
    .reset_index()
)


# ==============================
# 8. FILTER REAL PRODUCTS
# ==============================

filtered_products = product_summary[
    (product_summary["Months"] >= 12) &
    (product_summary["Total_Quantity"] >= 500) &
    (product_summary["Total_Transactions"] >= 50)
].copy()


# ==============================
# 9. PRICE VARIATION
# ==============================

filtered_products["Price_Range"] = (
    filtered_products["Max_Price"]
    -
    filtered_products["Min_Price"]
)

filtered_products["Price_Variation_Pct"] = (
    filtered_products["Price_Range"]
    /
    filtered_products["Avg_Price"]
    * 100
)


# ==============================
# 10. REQUIRE PRICE MOVEMENT
# ==============================

filtered_products = filtered_products[
    filtered_products["Price_Variation_Pct"] >= 5
].copy()


# ==============================
# 11. SORT BY REVENUE
# ==============================

filtered_products = (
    filtered_products
    .sort_values(
        "Total_Revenue",
        ascending=False
    )
    .reset_index(drop=True)
)


# ==============================
# 12. SHOW TOP PRODUCTS
# ==============================

print("\nFiltered products:", len(filtered_products))

print("\nTop 20 products:\n")

print(
    filtered_products[
        [
            "StockCode",
            "Description",
            "Months",
            "Total_Quantity",
            "Total_Revenue",
            "Avg_Price",
            "Min_Price",
            "Max_Price",
            "Price_Variation_Pct"
        ]
    ]
    .head(20)
    .to_string(index=False)
)