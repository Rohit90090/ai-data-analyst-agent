from pathlib import Path
import pandas as pd

# Project root
BASE_DIR = Path(__file__).resolve().parents[1]

# Clean dataset
input_file = BASE_DIR / "data" / "processed" / "online_retail_cleaned.csv"

print("Loading cleaned dataset...")

df = pd.read_csv(input_file)

# Convert date
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print("\nDataset loaded successfully!")

# ------------------------------------------------
# 1. Overall KPIs
# ------------------------------------------------

total_revenue = df["Revenue"].sum()
total_orders = df["Invoice"].nunique()
total_customers = df["Customer ID"].nunique()
total_products = df["StockCode"].nunique()
total_quantity = df["Quantity"].sum()

average_order_value = total_revenue / total_orders

print("\n========== BUSINESS KPIs ==========")

print(f"Total Revenue: £{total_revenue:,.2f}")
print(f"Total Orders: {total_orders:,}")
print(f"Total Customers: {total_customers:,}")
print(f"Total Products: {total_products:,}")
print(f"Total Quantity Sold: {total_quantity:,}")
print(f"Average Order Value: £{average_order_value:,.2f}")

# ------------------------------------------------
# 2. Monthly Revenue
# ------------------------------------------------

df["Month"] = df["InvoiceDate"].dt.to_period("M")

monthly_revenue = (
    df.groupby("Month")["Revenue"]
    .sum()
    .sort_index()
)

print("\n========== MONTHLY REVENUE ==========")
print(monthly_revenue)

# ------------------------------------------------
# 3. Top 10 Products by Revenue
# ------------------------------------------------

top_products = (
    df.groupby("Description")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n========== TOP 10 PRODUCTS ==========")
print(top_products)

# ------------------------------------------------
# 4. Top 10 Countries by Revenue
# ------------------------------------------------

top_countries = (
    df.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n========== TOP 10 COUNTRIES ==========")
print(top_countries)

# ------------------------------------------------
# 5. Customer Revenue
# ------------------------------------------------

customer_revenue = (
    df.groupby("Customer ID")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\n========== TOP 10 CUSTOMERS ==========")
print(customer_revenue.head(10))

# ------------------------------------------------
# 6. Orders per Customer
# ------------------------------------------------

orders_per_customer = (
    df.groupby("Customer ID")["Invoice"]
    .nunique()
)

print("\n========== CUSTOMER ORDER STATS ==========")

print(
    f"Average orders per customer: "
    f"{orders_per_customer.mean():.2f}"
)

print(
    f"Maximum orders by one customer: "
    f"{orders_per_customer.max()}"
)

# ------------------------------------------------
# 7. Revenue by Country
# ------------------------------------------------

country_summary = (
    df.groupby("Country")
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("Invoice", "nunique"),
        Customers=("Customer ID", "nunique")
    )
    .sort_values("Revenue", ascending=False)
)

print("\n========== COUNTRY SUMMARY ==========")
print(country_summary.head(10))

# ------------------------------------------------
# 8. Save analysis outputs
# ------------------------------------------------

output_dir = BASE_DIR / "data" / "processed"

monthly_revenue.to_csv(
    output_dir / "monthly_revenue.csv"
)

top_products.to_csv(
    output_dir / "top_products.csv"
)

country_summary.to_csv(
    output_dir / "country_summary.csv"
)

print("\nAnalysis files saved successfully!")