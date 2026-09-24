from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import joblib


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

input_file = BASE_DIR / "data" / "processed" / "product_monthly_demand.csv"
model_dir = BASE_DIR / "data" / "models"
output_dir = BASE_DIR / "data" / "processed"

model_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading monthly demand dataset...")

df = pd.read_csv(input_file)

print(f"Rows loaded: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 3. DATE PROCESSING
# ============================================================

df["Month"] = pd.to_datetime(df["Month"])

df = df.sort_values(["StockCode", "Month"]).reset_index(drop=True)


# ============================================================
# 4. CREATE TIME FEATURES
# ============================================================

df["Year"] = df["Month"].dt.year
df["Month_Number"] = df["Month"].dt.month

# Seasonal features
df["Month_Sin"] = np.sin(2 * np.pi * df["Month_Number"] / 12)
df["Month_Cos"] = np.cos(2 * np.pi * df["Month_Number"] / 12)


# ============================================================
# 5. CREATE HISTORICAL DEMAND FEATURES
# ============================================================

print("\nCreating historical demand features...")

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

# Rolling demand
df["Quantity_3M_Avg"] = (
    df.groupby("StockCode")["Quantity"]
    .transform(lambda x: x.shift(1).rolling(3).mean())
)

# Price change
df["Previous_Price"] = (
    df.groupby("StockCode")["Avg_Price"]
    .shift(1)
)

df["Price_Change_Pct"] = (
    (df["Avg_Price"] - df["Previous_Price"])
    / df["Previous_Price"]
) * 100


# ============================================================
# 6. REMOVE ROWS WITHOUT HISTORY
# ============================================================

df = df.dropna().reset_index(drop=True)

print(f"Rows after feature creation: {len(df):,}")


# ============================================================
# 7. PRODUCT ENCODING
# ============================================================

print("\nEncoding products...")

encoder = LabelEncoder()

df["Product_Code"] = encoder.fit_transform(
    df["StockCode"].astype(str)
)


# ============================================================
# 8. SELECT FEATURES
# ============================================================

features = [
    "Product_Code",
    "Avg_Price",
    "Previous_Price",
    "Price_Change_Pct",
    "Previous_Quantity",
    "Previous_Revenue",
    "Previous_Customers",
    "Previous_Transactions",
    "Quantity_3M_Avg",
    "Month_Number",
    "Month_Sin",
    "Month_Cos"
]

target = "Quantity"

X = df[features]
y = df[target]


# ============================================================
# 9. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

print("\nCreating time-based train/test split...")

# Important:
# We DO NOT randomly split time-series data.

split_date = df["Month"].quantile(0.80)

train_mask = df["Month"] <= split_date
test_mask = df["Month"] > split_date

X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]

print(f"Train rows: {len(X_train):,}")
print(f"Test rows : {len(X_test):,}")

print(f"Train until: {df.loc[train_mask, 'Month'].max()}")
print(f"Test from  : {df.loc[test_mask, 'Month'].min()}")


# ============================================================
# 10. MODEL 1 — LINEAR REGRESSION
# ============================================================

print("\nTraining Linear Regression...")

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_pred)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
linear_r2 = r2_score(y_test, linear_pred)


# ============================================================
# 11. MODEL 2 — RANDOM FOREST
# ============================================================

print("Training Random Forest...")

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)


# ============================================================
# 12. MODEL 3 — GRADIENT BOOSTING
# ============================================================

print("Training Gradient Boosting...")

gb_model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_pred = gb_model.predict(X_test)

gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_r2 = r2_score(y_test, gb_pred)


# ============================================================
# 13. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "MAE": [
        linear_mae,
        rf_mae,
        gb_mae
    ],
    "RMSE": [
        linear_rmse,
        rf_rmse,
        gb_rmse
    ],
    "R2": [
        linear_r2,
        rf_r2,
        gb_r2
    ]
})

results = results.sort_values("MAE")

print("\n==============================================")
print("          MODEL PERFORMANCE")
print("==============================================")

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 14. SELECT BEST MODEL
# ============================================================

best_model_name = results.iloc[0]["Model"]

if best_model_name == "Linear Regression":
    best_model = linear_model

elif best_model_name == "Random Forest":
    best_model = rf_model

else:
    best_model = gb_model


print("\nBest model:", best_model_name)


# ============================================================
# 15. SAVE MODEL
# ============================================================

model_file = model_dir / "best_demand_model.pkl"
encoder_file = model_dir / "product_encoder.pkl"
features_file = model_dir / "model_features.pkl"

joblib.dump(best_model, model_file)
joblib.dump(encoder, encoder_file)
joblib.dump(features, features_file)

print("\nModel saved:")
print(model_file)


# ============================================================
# 16. SAVE MODEL RESULTS
# ============================================================

results_file = output_dir / "model_comparison.csv"

results.to_csv(
    results_file,
    index=False
)

print("\nModel comparison saved:")
print(results_file)


# ============================================================
# 17. SAVE TEST PREDICTIONS
# ============================================================

test_results = df.loc[test_mask].copy()

test_results["Actual_Quantity"] = y_test.values
test_results["Predicted_Quantity"] = rf_pred if best_model_name == "Random Forest" else (
    linear_pred if best_model_name == "Linear Regression" else gb_pred
)

test_results["Prediction_Error"] = (
    test_results["Actual_Quantity"]
    - test_results["Predicted_Quantity"]
)

prediction_file = output_dir / "demand_predictions.csv"

test_results.to_csv(
    prediction_file,
    index=False
)

print("\nPredictions saved:")
print(prediction_file)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n==============================================")
print("             STEP 8 COMPLETE")
print("==============================================")

print(f"Best Model : {best_model_name}")
print(f"MAE        : {results.iloc[0]['MAE']:.4f}")
print(f"RMSE       : {results.iloc[0]['RMSE']:.4f}")
print(f"R²         : {results.iloc[0]['R2']:.4f}")

print("\nFiles created:")
print("- best_demand_model.pkl")
print("- product_encoder.pkl")
print("- model_features.pkl")
print("- model_comparison.csv")
print("- demand_predictions.csv")