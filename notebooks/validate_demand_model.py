from pathlib import Path
import pandas as pd
import numpy as np
import joblib


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

data_file = (
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

output_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "price_scenario_results.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading monthly demand dataset...")

df = pd.read_csv(data_file)

df["Month"] = pd.to_datetime(df["Month"])

print(f"Rows loaded: {len(df):,}")


# ============================================================
# 3. RECREATE MODEL FEATURES
# ============================================================

print("\nCreating historical features...")

df = df.sort_values(
    ["StockCode", "Month"]
).reset_index(drop=True)

df["Year"] = df["Month"].dt.year
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
        lambda x: x.shift(1).rolling(3).mean()
    )
)

df["Previous_Price"] = (
    df.groupby("StockCode")["Avg_Price"]
    .shift(1)
)

df["Price_Change_Pct"] = (
    (df["Avg_Price"] - df["Previous_Price"])
    / df["Previous_Price"]
) * 100


# ============================================================
# 4. REMOVE INVALID HISTORY
# ============================================================

df_model = df.dropna().copy()

print(
    f"Rows available for simulation: {len(df_model):,}"
)


# ============================================================
# 5. LOAD MODEL
# ============================================================

print("\nLoading saved model...")

model = joblib.load(model_file)
encoder = joblib.load(encoder_file)
features = joblib.load(features_file)

print("Model loaded successfully.")
print(f"Model features: {len(features)}")


# ============================================================
# 6. SELECT PRODUCTS
# ============================================================

product_summary = (
    df_model.groupby(
        ["StockCode", "Description"]
    )
    .agg(
        Total_Quantity=("Quantity", "sum"),
        Total_Revenue=("Revenue", "sum"),
        Avg_Price=("Avg_Price", "mean"),
        Months=("Month", "nunique")
    )
    .reset_index()
)

# Products with enough history
product_summary = product_summary[
    product_summary["Months"] >= 6
]

# Select top revenue products
selected_products = (
    product_summary
    .sort_values(
        "Total_Revenue",
        ascending=False
    )
    .head(5)
)

print("\n==============================================")
print("SELECTED PRODUCTS")
print("==============================================")

print(
    selected_products[
        [
            "StockCode",
            "Description",
            "Total_Quantity",
            "Total_Revenue",
            "Avg_Price",
            "Months"
        ]
    ].to_string(index=False)
)


# ============================================================
# 7. SCENARIO FUNCTION
# ============================================================

def simulate_price_scenario(
    product_code,
    new_price
):

    product_rows = df_model[
        df_model["StockCode"] == product_code
    ].sort_values("Month")

    if len(product_rows) == 0:
        return None

    # Latest historical observation
    latest = product_rows.iloc[-1]

    current_price = float(
        latest["Avg_Price"]
    )

    current_quantity = float(
        latest["Quantity"]
    )

    current_revenue = float(
        latest["Revenue"]
    )

    # Price change
    price_change_pct = (
        (new_price - current_price)
        / current_price
    ) * 100

    # Encode product
    encoded_product = encoder.transform(
        [str(product_code)]
    )[0]

    # Build scenario
    scenario = {
        "Product_Code": encoded_product,

        "Avg_Price": new_price,

        "Previous_Price": current_price,

        "Price_Change_Pct": price_change_pct,

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

    X_scenario = pd.DataFrame(
        [scenario]
    )[features]

    # Prediction
    predicted_quantity = float(
        model.predict(X_scenario)[0]
    )

    # Demand cannot be negative
    predicted_quantity = max(
        0,
        predicted_quantity
    )

    # Revenue
    predicted_revenue = (
        predicted_quantity
        * new_price
    )

    revenue_change = (
        predicted_revenue
        - current_revenue
    )

    revenue_change_pct = (
        revenue_change
        / current_revenue
        * 100
        if current_revenue > 0
        else 0
    )

    return {
        "StockCode": product_code,

        "Description":
            latest["Description"],

        "Reference_Month":
            latest["Month"],

        "Current_Price":
            current_price,

        "New_Price":
            new_price,

        "Price_Change_Pct":
            price_change_pct,

        "Current_Quantity":
            current_quantity,

        "Predicted_Quantity":
            predicted_quantity,

        "Current_Revenue":
            current_revenue,

        "Predicted_Revenue":
            predicted_revenue,

        "Revenue_Change":
            revenue_change,

        "Revenue_Change_Pct":
            revenue_change_pct
    }


# ============================================================
# 8. RUN SCENARIOS
# ============================================================

print("\n==============================================")
print("RUNNING PRICE SCENARIOS")
print("==============================================")

results = []

for _, product in selected_products.iterrows():

    product_code = product["StockCode"]

    product_rows = df_model[
        df_model["StockCode"]
        == product_code
    ]

    latest_price = (
        product_rows
        .sort_values("Month")
        .iloc[-1]["Avg_Price"]
    )

    scenario_prices = [
        latest_price * 0.80,
        latest_price * 0.90,
        latest_price,
        latest_price * 1.10,
        latest_price * 1.20
    ]

    for price in scenario_prices:

        result = simulate_price_scenario(
            product_code,
            price
        )

        if result is not None:
            results.append(result)


# ============================================================
# 9. RESULTS DATAFRAME
# ============================================================

if len(results) == 0:

    print("\nERROR: No scenario results generated.")

    raise SystemExit


results_df = pd.DataFrame(results)

numeric_columns = [
    "Current_Price",
    "New_Price",
    "Price_Change_Pct",
    "Current_Quantity",
    "Predicted_Quantity",
    "Current_Revenue",
    "Predicted_Revenue",
    "Revenue_Change",
    "Revenue_Change_Pct"
]

results_df[numeric_columns] = (
    results_df[numeric_columns]
    .round(2)
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n==============================================")
print("PRICE SCENARIO RESULTS")
print("==============================================")


for product in results_df[
    "Description"
].unique():

    print(
        f"\nPRODUCT: {product}"
    )

    product_results = (
        results_df[
            results_df["Description"]
            == product
        ]
        .sort_values("New_Price")
    )

    print(
        product_results[
            [
                "Reference_Month",
                "Current_Price",
                "New_Price",
                "Price_Change_Pct",
                "Current_Quantity",
                "Predicted_Quantity",
                "Predicted_Revenue",
                "Revenue_Change_Pct"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 11. DEMAND BEHAVIOR CHECK
# ============================================================

print("\n==============================================")
print("DEMAND BEHAVIOR CHECK")
print("==============================================")

behavior_results = []

for product in results_df[
    "Description"
].unique():

    product_results = (
        results_df[
            results_df["Description"]
            == product
        ]
        .sort_values("New_Price")
    )

    quantities = (
        product_results[
            "Predicted_Quantity"
        ]
        .values
    )

    demand_decreases = 0

    comparisons = len(quantities) - 1

    for i in range(
        1,
        len(quantities)
    ):

        if quantities[i] <= quantities[i - 1]:

            demand_decreases += 1

    rate = (
        demand_decreases
        / comparisons
        if comparisons > 0
        else 0
    )

    behavior_results.append({
        "Description": product,

        "Price_Levels":
            len(quantities),

        "Demand_Decrease_Rate":
            round(rate, 2)
    })


behavior_df = pd.DataFrame(
    behavior_results
)

print(
    behavior_df.to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

results_df.to_csv(
    output_file,
    index=False
)

print("\n==============================================")
print("STEP 9 COMPLETE")
print("==============================================")

print(
    f"\nTotal scenarios: {len(results_df)}"
)

print("\nSaved to:")

print(output_file)