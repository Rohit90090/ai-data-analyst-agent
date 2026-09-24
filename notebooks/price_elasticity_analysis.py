from pathlib import Path
import pandas as pd
import numpy as np

# Project root
BASE_DIR = Path(__file__).resolve().parents[1]

# Input
input_file = BASE_DIR / "data" / "processed" / "online_retail_cleaned.csv"

print("Loading cleaned dataset...")

df = pd.read_csv(input_file)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# ------------------------------------------------
# Selected products
# ------------------------------------------------

selected_products = [
    "WHITE HANGING HEART T-LIGHT HOLDER",
    "REGENCY CAKESTAND 3 TIER",
    "ASSORTED COLOUR BIRD ORNAMENT",
    "JUMBO BAG RED RETROSPOT",
    "PARTY BUNTING"
]

df = df[df["Description"].isin(selected_products)].copy()

print(f"\nRows for selected products: {len(df):,}")

# ------------------------------------------------
# Aggregate by product + price
# ------------------------------------------------

price_demand = (
    df.groupby(["Description", "Price"])
    .agg(
        Quantity=("Quantity", "sum"),
        Transactions=("Invoice", "count"),
        Revenue=("Revenue", "sum")
    )
    .reset_index()
)

# ------------------------------------------------
# Calculate elasticity
# Elasticity = % change in quantity / % change in price
# ------------------------------------------------

results = []

for product in selected_products:

    product_data = price_demand[
        price_demand["Description"] == product
    ].sort_values("Price").copy()

    if len(product_data) < 2:
        continue

    product_data["Price_Pct_Change"] = (
        product_data["Price"].pct_change()
    )

    product_data["Quantity_Pct_Change"] = (
        product_data["Quantity"].pct_change()
    )

    product_data["Elasticity"] = (
        product_data["Quantity_Pct_Change"] /
        product_data["Price_Pct_Change"]
    )

    # Remove infinite values
    elasticity_values = product_data["Elasticity"].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(elasticity_values) > 0:

        median_elasticity = elasticity_values.median()

        results.append({
            "Product": product,
            "Price_Points": len(product_data),
            "Median_Elasticity": median_elasticity,
            "Min_Price": product_data["Price"].min(),
            "Max_Price": product_data["Price"].max(),
            "Total_Quantity": product_data["Quantity"].sum(),
            "Total_Revenue": product_data["Revenue"].sum()
        })

# ------------------------------------------------
# Results
# ------------------------------------------------

elasticity_df = pd.DataFrame(results)

print("\n========== PRICE ELASTICITY RESULTS ==========")

print(
    elasticity_df.to_string(index=False)
)

# ------------------------------------------------
# Save results
# ------------------------------------------------

output_file = (
    BASE_DIR /
    "data" /
    "processed" /
    "price_elasticity_results.csv"
)

elasticity_df.to_csv(
    output_file,
    index=False
)

print("\nResults saved to:")
print(output_file)