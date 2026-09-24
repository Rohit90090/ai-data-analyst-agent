from pathlib import Path
import pandas as pd
import numpy as np
import joblib


# ==============================
# 1. PATHS
# ==============================

BASE_DIR = Path(__file__).resolve().parents[1]

demand_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "product_monthly_demand.csv"
)

model_file = (
    BASE_DIR
    / "data"
    / "models"
    / "best_demand_model.pkl"
)

encoder_file = (
    BASE_DIR
    / "data"
    / "models"
    / "product_encoder.pkl"
)

features_file = (
    BASE_DIR
    / "data"
    / "models"
    / "model_features.pkl"
)


# ==============================
# 2. LOAD
# ==============================

df = pd.read_csv(demand_file)

df["Month"] = pd.to_datetime(df["Month"])

model = joblib.load(model_file)
encoder = joblib.load(encoder_file)
features = joblib.load(features_file)


# ==============================
# 3. FILTER REAL PRODUCTS
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
    .astype(str)
    .str.upper()
    .isin(excluded_codes)
].copy()


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
    .fillna("")
    .str.upper()
    .str.contains(
        pattern,
        regex=True,
        na=False
    )
].copy()


# ==============================
# 4. CREATE FEATURES
# ==============================

df = df.sort_values(
    ["StockCode", "Month"]
).reset_index(drop=True)

df["Month_Number"] = df["Month"].dt.month

df["Month_Sin"] = np.sin(
    2 * np.pi * df["Month_Number"] / 12
)

df["Month_Cos"] = np.cos(
    2 * np.pi * df["Month_Number"] / 12
)

df["Previous_Quantity"] = (
    df.groupby("StockCode")["Quantity"]
    .shift(1)
)

df["Previous_Revenue"] = (
    df.groupby("StockCode")["Revenue"]
    .shift(1)
)

df["Previous_Customers"] = (
    df.groupby("StockCode")["Customers"]
    .shift(1)
)

df["Previous_Transactions"] = (
    df.groupby("StockCode")["Transactions"]
    .shift(1)
)

df["Quantity_3M_Avg"] = (
    df.groupby("StockCode")["Quantity"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(3)
        .mean()
    )
)

df["Previous_Price"] = (
    df.groupby("StockCode")["Avg_Price"]
    .shift(1)
)

df["Price_Change_Pct"] = (
    (
        df["Avg_Price"]
        - df["Previous_Price"]
    )
    / df["Previous_Price"]
) * 100


df = df.dropna().copy()


# ==============================
# 5. PRODUCT HISTORY FILTER
# ==============================

product_history = (
    df.groupby("StockCode")
    .agg(
        Months=("Month", "nunique"),
        Total_Quantity=("Quantity", "sum"),
        Total_Revenue=("Revenue", "sum")
    )
    .reset_index()
)

valid_products = product_history[
    (product_history["Months"] >= 12) &
    (product_history["Total_Quantity"] >= 500)
]["StockCode"]


df = df[
    df["StockCode"].isin(valid_products)
].copy()


# ==============================
# 6. SELECT PRODUCTS
# ==============================

top_products = (
    product_history[
        product_history["StockCode"]
        .isin(valid_products)
    ]
    .sort_values(
        "Total_Revenue",
        ascending=False
    )
    .head(10)["StockCode"]
    .tolist()
)


# ==============================
# 7. PRICE SCENARIOS
# ==============================

price_changes = [
    -20,
    -10,
    0,
    10,
    20
]

results = []


# ==============================
# 8. RUN SANITY CHECK
# ==============================

for product_code in top_products:

    product_data = df[
        df["StockCode"] == product_code
    ].sort_values("Month")

    if len(product_data) == 0:
        continue

    latest = product_data.iloc[-1]

    current_price = float(
        latest["Avg_Price"]
    )

    historical_quantities = (
        product_data["Quantity"]
    )

    historical_median = float(
        historical_quantities.median()
    )

    historical_max = float(
        historical_quantities.quantile(0.95)
    )

    for change in price_changes:

        new_price = (
            current_price
            * (1 + change / 100)
        )

        encoded_product = encoder.transform(
            [str(product_code)]
        )[0]

        scenario = {
            "Product_Code":
                encoded_product,

            "Avg_Price":
                new_price,

            "Previous_Price":
                current_price,

            "Price_Change_Pct":
                change,

            "Previous_Quantity":
                latest["Previous_Quantity"],

            "Previous_Revenue":
                latest["Previous_Revenue"],

            "Previous_Customers":
                latest["Previous_Customers"],

            "Previous_Transactions":
                latest["Previous_Transactions"],

            "Quantity_3M_Avg":
                latest["Quantity_3M_Avg"],

            "Month_Number":
                latest["Month_Number"],

            "Month_Sin":
                latest["Month_Sin"],

            "Month_Cos":
                latest["Month_Cos"]
        }

        X = pd.DataFrame(
            [scenario]
        )[features]

        raw_prediction = float(
            model.predict(X)[0]
        )

        # Historical demand based limit
        upper_limit = max(
            historical_max,
            latest["Quantity_3M_Avg"] * 2,
            latest["Previous_Quantity"] * 2,
            1
        )

        constrained_prediction = np.clip(
            raw_prediction,
            0,
            upper_limit
        )

        predicted_revenue = (
            constrained_prediction
            * new_price
        )

        results.append({
            "StockCode":
                product_code,

            "Description":
                latest["Description"],

            "Reference_Month":
                latest["Month"],

            "Current_Price":
                round(current_price, 2),

            "New_Price":
                round(new_price, 2),

            "Price_Change_Pct":
                change,

            "Historical_Median_Quantity":
                round(historical_median, 2),

            "Historical_95th_Quantity":
                round(historical_max, 2),

            "Raw_Model_Quantity":
                round(raw_prediction, 2),

            "Constrained_Quantity":
                round(
                    constrained_prediction,
                    2
                ),

            "Predicted_Revenue":
                round(
                    predicted_revenue,
                    2
                )
        })


# ==============================
# 9. DISPLAY RESULTS
# ==============================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 100)
print("DEMAND MODEL SANITY CHECK")
print("=" * 100)

print(
    results_df[
        [
            "Description",
            "Price_Change_Pct",
            "Current_Price",
            "New_Price",
            "Historical_Median_Quantity",
            "Raw_Model_Quantity",
            "Constrained_Quantity",
            "Predicted_Revenue"
        ]
    ].to_string(index=False)
)


# ==============================
# 10. EXTREME PREDICTION CHECK
# ==============================

print("\n")
print("=" * 100)
print("EXTREME PREDICTION CHECK")
print("=" * 100)

print(
    "Maximum raw prediction:",
    round(
        results_df["Raw_Model_Quantity"].max(),
        2
    )
)

print(
    "Maximum constrained prediction:",
    round(
        results_df["Constrained_Quantity"].max(),
        2
    )
)

print(
    "Minimum constrained prediction:",
    round(
        results_df["Constrained_Quantity"].min(),
        2
    )
)