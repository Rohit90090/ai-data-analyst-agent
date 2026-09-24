from pathlib import Path
import pandas as pd

# Project root
BASE_DIR = Path(__file__).resolve().parents[1]

# Clean dataset
input_file = BASE_DIR / "data" / "processed" / "online_retail_cleaned.csv"

print("Loading cleaned dataset...")

df = pd.read_csv(input_file)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print(f"Rows loaded: {len(df):,}")

# ------------------------------------------------
# 1. Product-level price variation
# ------------------------------------------------

product_stats = (
    df.groupby(["StockCode", "Description"])
    .agg(
        Transactions=("Invoice", "count"),
        Unique_Prices=("Price", "nunique"),
        Min_Price=("Price", "min"),
        Max_Price=("Price", "max"),
        Avg_Price=("Price", "mean"),
        Total_Quantity=("Quantity", "sum"),
        Total_Revenue=("Revenue", "sum")
    )
    .reset_index()
)

# ------------------------------------------------
# 2. Calculate price range
# ------------------------------------------------

product_stats["Price_Range"] = (
    product_stats["Max_Price"] -
    product_stats["Min_Price"]
)

product_stats["Price_Variation_%"] = (
    product_stats["Price_Range"] /
    product_stats["Avg_Price"]
) * 100

# ------------------------------------------------
# 3. Find products suitable for simulation
# ------------------------------------------------

suitable_products = product_stats[
    (product_stats["Transactions"] >= 100) &
    (product_stats["Unique_Prices"] >= 3) &
    (product_stats["Price_Range"] > 0)
].copy()

# Sort by transaction count
suitable_products = suitable_products.sort_values(
    "Transactions",
    ascending=False
)

print("\n========== PRODUCT PRICE ANALYSIS ==========")

print(
    f"Total unique products: "
    f"{len(product_stats):,}"
)

print(
    f"Products with multiple prices: "
    f"{(product_stats['Unique_Prices'] > 1).sum():,}"
)

print(
    f"Products suitable for simulation: "
    f"{len(suitable_products):,}"
)

# ------------------------------------------------
# 4. Display top candidates
# ------------------------------------------------

print("\n========== TOP SIMULATION CANDIDATES ==========")

print(
    suitable_products[
        [
            "StockCode",
            "Description",
            "Transactions",
            "Unique_Prices",
            "Min_Price",
            "Max_Price",
            "Avg_Price",
            "Price_Variation_%",
            "Total_Quantity",
            "Total_Revenue"
        ]
    ].head(20).to_string(index=False)
)

# ------------------------------------------------
# 5. Save results
# ------------------------------------------------

output_file = (
    BASE_DIR /
    "data" /
    "processed" /
    "price_demand_candidates.csv"
)

suitable_products.to_csv(
    output_file,
    index=False
)

print("\nAnalysis completed!")

print("Saved to:")
print(output_file)