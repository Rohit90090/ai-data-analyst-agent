from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "product_monthly_demand.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "data"
    / "models"
    / "best_demand_model.pkl"
)

ENCODER_PATH = (
    BASE_DIR
    / "data"
    / "models"
    / "product_encoder.pkl"
)

FEATURES_PATH = (
    BASE_DIR
    / "data"
    / "models"
    / "model_features.pkl"
)


# ============================================================
# LOAD DATA + MODELS
# ============================================================

print("\nLoading data and models...")

df = pd.read_csv(DATA_PATH)

df["Month"] = pd.to_datetime(df["Month"])

ml_model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
model_features = joblib.load(FEATURES_PATH)

print(f"Monthly demand rows: {len(df):,}")


# ============================================================
# REMOVE NON-PRODUCT / ADMIN ITEMS
# ============================================================

excluded_codes = {
    "POST",
    "DOT",
    "M",
    "C2",
    "BANK CHARGES",
    "D",
    "S",
    "CRUK"
}

excluded_words = [
    "POSTAGE",
    "DISCOUNT",
    "MANUAL",
    "BANK CHARGE",
    "AMAZON FEE",
    "CRUK"
]

df["StockCode"] = df["StockCode"].astype(str)

df = df[
    ~df["StockCode"]
    .str.upper()
    .isin(excluded_codes)
].copy()

description_upper = (
    df["Description"]
    .fillna("")
    .astype(str)
    .str.upper()
)

for word in excluded_words:
    df = df[
        ~description_upper.str.contains(
            word,
            na=False
        )
    ]


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

df = df[
    (df["Quantity"] > 0)
    & (df["Avg_Price"] > 0)
].copy()

df = df.sort_values(
    ["StockCode", "Month"]
)


# ============================================================
# ROBUST PRICE SENSITIVITY
#
# Instead of:
#
# log(Q) ~ log(P) + month dummies
#
# we estimate:
#
# change in log(Q) ~ change in log(P)
#
# This avoids the previous artificial R² = 1.0 issue.
# ============================================================

print("\nEstimating historical price response...\n")

elasticity_results = []


for stock_code, group in df.groupby("StockCode"):

    group = (
        group
        .sort_values("Month")
        .copy()
    )

    # Require enough monthly observations
    if len(group) < 12:
        continue

    # --------------------------------------------------------
    # Log transformation
    # --------------------------------------------------------

    group["Log_Quantity"] = np.log1p(
        group["Quantity"]
    )

    group["Log_Price"] = np.log(
        group["Avg_Price"]
    )

    # --------------------------------------------------------
    # Month-to-month changes
    # --------------------------------------------------------

    group["Delta_Log_Quantity"] = (
        group["Log_Quantity"].diff()
    )

    group["Delta_Log_Price"] = (
        group["Log_Price"].diff()
    )

    changes = group[
        group["Delta_Log_Quantity"].notna()
        &
        group["Delta_Log_Price"].notna()
    ].copy()

    # Remove almost-zero price changes
    changes = changes[
        changes["Delta_Log_Price"].abs() >= 0.01
    ]

    if len(changes) < 8:
        continue

    X = changes[
        ["Delta_Log_Price"]
    ]

    y = changes[
        "Delta_Log_Quantity"
    ]

    regression = LinearRegression()

    regression.fit(
        X,
        y
    )

    elasticity = float(
        regression.coef_[0]
    )

    r2 = float(
        regression.score(
            X,
            y
        )
    )

    # Number of meaningful price changes
    meaningful_changes = int(
        (
            changes["Delta_Log_Price"].abs()
            >= 0.02
        ).sum()
    )

    elasticity_results.append({

        "StockCode": stock_code,

        "Description": group[
            "Description"
        ].iloc[-1],

        "Elasticity": elasticity,

        "Elasticity_R2": r2,

        "Observations": len(group),

        "Price_Change_Pairs": len(changes),

        "Meaningful_Price_Changes":
            meaningful_changes

    })


elasticity_df = pd.DataFrame(
    elasticity_results
)


# ============================================================
# ELASTICITY QUALITY CLASSIFICATION
# ============================================================

def classify_elasticity(row):

    elasticity = row["Elasticity"]

    r2 = row["Elasticity_R2"]

    pairs = row["Price_Change_Pairs"]

    meaningful = (
        row["Meaningful_Price_Changes"]
    )

    if pairs < 8:
        return "Insufficient Evidence"

    if meaningful < 3:
        return "Insufficient Price Variation"

    if elasticity >= 0:
        return "Unreliable Price Response"

    # Very large elasticity estimates are unstable
    if elasticity < -5:
        return "High Sensitivity"

    if r2 < 0.05:
        return "Weak Relationship"

    if r2 < 0.15:
        return "Moderate Evidence"

    return "Usable Evidence"


elasticity_df[
    "Elasticity_Status"
] = elasticity_df.apply(
    classify_elasticity,
    axis=1
)


# ============================================================
# ELASTICITY SUMMARY
# ============================================================

print("=" * 110)
print("ELASTICITY QUALITY SUMMARY")
print("=" * 110)

print(
    elasticity_df[
        [
            "Description",
            "Elasticity",
            "Elasticity_R2",
            "Observations",
            "Price_Change_Pairs",
            "Elasticity_Status"
        ]
    ]
    .sort_values(
        "Elasticity_R2",
        ascending=False
    )
    .head(20)
    .round(3)
    .to_string(index=False)
)


# ============================================================
# DEMAND MODEL FEATURE RECREATION
#
# IMPORTANT:
# These are the features used by the already-trained
# demand model.
# ============================================================

def create_model_features(product_df):

    data = (
        product_df
        .sort_values("Month")
        .copy()
    )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    data["Month_Number"] = (
        data["Month"].dt.month
    )

    data["Month_Sin"] = np.sin(
        2
        * np.pi
        * data["Month_Number"]
        / 12
    )

    data["Month_Cos"] = np.cos(
        2
        * np.pi
        * data["Month_Number"]
        / 12
    )

    data["Year"] = (
        data["Month"].dt.year
    )

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    data["Previous_Price"] = (
        data["Avg_Price"].shift(1)
    )

    data["Previous_Quantity"] = (
        data["Quantity"].shift(1)
    )

    data["Previous_Customers"] = (
        data["Customers"].shift(1)
    )

    data["Previous_Transactions"] = (
        data["Transactions"].shift(1)
    )

    return data


# ============================================================
# ML DEMAND PREDICTION
# ============================================================

def predict_ml_demand(product_df):

    data = create_model_features(
        product_df
    )

    # --------------------------------------------------------
    # Use latest observation
    # --------------------------------------------------------

    row = data.iloc[-1].copy()

    # --------------------------------------------------------
    # Fill lag features if latest row has missing values
    # --------------------------------------------------------

    numeric_columns = [
        "Avg_Price",
        "Min_Price",
        "Max_Price",
        "Revenue_Per_Unit",
        "Transactions",
        "Customers",
        "Quantity",
        "Revenue",
        "Previous_Price",
        "Previous_Quantity",
        "Previous_Customers",
        "Previous_Transactions",
        "Month_Number",
        "Month_Sin",
        "Month_Cos",
        "Year"
    ]

    for column in numeric_columns:

        if column in row.index:

            if pd.isna(row[column]):

                valid_values = data[
                    column
                ].dropna()

                if len(valid_values) > 0:

                    row[column] = (
                        valid_values.iloc[-1]
                    )

    # --------------------------------------------------------
    # Encode product
    # --------------------------------------------------------

    try:

        encoded = encoder.transform(
            np.array(
                [row["StockCode"]]
            )
        )[0]

    except Exception:

        # If encoder cannot process product,
        # use recent actual demand as fallback.
        return float(
            product_df[
                "Quantity"
            ]
            .tail(3)
            .mean()
        )

    # --------------------------------------------------------
    # Create complete feature dictionary
    # --------------------------------------------------------

    feature_values = {

        "Product_Encoded":
            encoded,

        "Avg_Price":
            row.get("Avg_Price", 0),

        "Min_Price":
            row.get("Min_Price", 0),

        "Max_Price":
            row.get("Max_Price", 0),

        "Revenue_Per_Unit":
            row.get(
                "Revenue_Per_Unit",
                0
            ),

        "Transactions":
            row.get(
                "Transactions",
                0
            ),

        "Customers":
            row.get(
                "Customers",
                0
            ),

        "Quantity":
            row.get(
                "Quantity",
                0
            ),

        "Revenue":
            row.get(
                "Revenue",
                0
            ),

        "Previous_Price":
            row.get(
                "Previous_Price",
                row.get(
                    "Avg_Price",
                    0
                )
            ),

        "Previous_Quantity":
            row.get(
                "Previous_Quantity",
                row.get(
                    "Quantity",
                    0
                )
            ),

        "Previous_Customers":
            row.get(
                "Previous_Customers",
                row.get(
                    "Customers",
                    0
                )
            ),

        "Previous_Transactions":
            row.get(
                "Previous_Transactions",
                row.get(
                    "Transactions",
                    0
                )
            ),

        "Month_Number":
            row.get(
                "Month_Number",
                row["Month"].month
            ),

        "Month_Sin":
            row.get(
                "Month_Sin",
                np.sin(
                    2
                    * np.pi
                    * row["Month"].month
                    / 12
                )
            ),

        "Month_Cos":
            row.get(
                "Month_Cos",
                np.cos(
                    2
                    * np.pi
                    * row["Month"].month
                    / 12
                )
            ),

        "Year":
            row.get(
                "Year",
                row["Month"].year
            )
    }

    feature_row = pd.DataFrame(
        [feature_values]
    )

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    feature_row = feature_row.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Fill missing values
    # --------------------------------------------------------

    for column in feature_row.columns:

        if feature_row[
            column
        ].isna().any():

            feature_row[
                column
            ] = feature_row[
                column
            ].fillna(0)

    # --------------------------------------------------------
    # Verify exact model features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in model_features
        if feature not in feature_row.columns
    ]

    if missing_features:

        print(
            "\nWARNING: Missing model features:"
        )

        print(
            missing_features
        )

        return float(
            product_df[
                "Quantity"
            ]
            .tail(3)
            .mean()
        )

    # --------------------------------------------------------
    # Exact feature order
    # --------------------------------------------------------

    X = feature_row[
        model_features
    ]

    prediction = float(
        ml_model.predict(X)[0]
    )

    return max(
        prediction,
        0
    )


# ============================================================
# SCENARIO ENGINE
# ============================================================

def generate_scenarios(stock_code):

    product = df[
        df["StockCode"]
        == stock_code
    ].sort_values(
        "Month"
    ).copy()

    if len(product) < 12:
        return None

    sensitivity = elasticity_df[
        elasticity_df["StockCode"]
        == stock_code
    ]

    if sensitivity.empty:
        return None

    sensitivity = (
        sensitivity
        .iloc[0]
    )

    elasticity = float(
        sensitivity["Elasticity"]
    )

    elasticity_r2 = float(
        sensitivity["Elasticity_R2"]
    )

    evidence = (
        sensitivity[
            "Elasticity_Status"
        ]
    )

    current_price = float(
        product.iloc[-1][
            "Avg_Price"
        ]
    )

    # --------------------------------------------------------
    # Recent actual demand
    # --------------------------------------------------------

    recent_demand = float(
        product
        .tail(3)[
            "Quantity"
        ]
        .mean()
    )

    # --------------------------------------------------------
    # ML demand prediction
    # --------------------------------------------------------

    ml_demand = predict_ml_demand(
        product
    )

    # --------------------------------------------------------
    # Blended baseline
    #
    # 60% recent actual
    # 40% ML prediction
    # --------------------------------------------------------

    base_demand = (
        recent_demand * 0.60
        +
        ml_demand * 0.40
    )

    scenarios = []

    price_changes = [
        -0.20,
        -0.10,
        0.00,
        0.10,
        0.20
    ]

    for price_change in price_changes:

        new_price = (
            current_price
            * (1 + price_change)
        )

        # ----------------------------------------------------
        # Demand calculation
        # ----------------------------------------------------

        if evidence in [
            "Usable Evidence",
            "Moderate Evidence"
        ] and elasticity < 0:

            price_ratio = (
                new_price
                / current_price
            )

            scenario_demand = (
                base_demand
                * (
                    price_ratio
                    ** elasticity
                )
            )

        else:

            # Weak/unreliable evidence:
            # keep demand at baseline instead
            # of making a misleading price response.
            scenario_demand = (
                base_demand
            )

        scenario_demand = max(
            scenario_demand,
            0
        )

        # ----------------------------------------------------
        # Revenue
        # ----------------------------------------------------

        revenue = (
            new_price
            * scenario_demand
        )

        baseline_revenue = (
            current_price
            * base_demand
        )

        demand_change_pct = (
            (
                scenario_demand
                / base_demand
            ) - 1
        ) * 100

        revenue_change_pct = (
            (
                revenue
                / baseline_revenue
            ) - 1
        ) * 100

        scenarios.append({

            "StockCode":
                stock_code,

            "Description":
                product.iloc[-1][
                    "Description"
                ],

            "Price_Change_Pct":
                price_change * 100,

            "Current_Price":
                current_price,

            "New_Price":
                new_price,

            "Base_Demand":
                base_demand,

            "ML_Base_Prediction":
                ml_demand,

            "Elasticity":
                elasticity,

            "Elasticity_R2":
                elasticity_r2,

            "Elasticity_Status":
                evidence,

            "Final_Demand":
                scenario_demand,

            "Predicted_Revenue":
                revenue,

            "Demand_Change_Pct":
                demand_change_pct,

            "Revenue_Change_Pct":
                revenue_change_pct
        })

    return pd.DataFrame(
        scenarios
    )


# ============================================================
# PRODUCT ELIGIBILITY
# ============================================================

eligible = (
    df.groupby("StockCode")
    .agg(
        Months=(
            "Month",
            "nunique"
        ),

        Total_Quantity=(
            "Quantity",
            "sum"
        ),

        Revenue=(
            "Revenue",
            "sum"
        )
    )
    .reset_index()
)

eligible = eligible[
    (eligible["Months"] >= 12)
    &
    (eligible["Total_Quantity"] >= 500)
]


# ============================================================
# SELECT RELIABLE PRODUCTS
#
# We deliberately exclude:
# - positive elasticity
# - high sensitivity below -5
# - insufficient evidence
#
# This makes the simulator more defensible.
# ============================================================

reliable_codes = set(
    elasticity_df[
        (
            elasticity_df[
                "Elasticity_Status"
            ].isin([
                "Usable Evidence",
                "Moderate Evidence"
            ])
        )
        &
        (
            elasticity_df[
                "Elasticity"
            ] < 0
        )
        &
        (
            elasticity_df[
                "Elasticity"
            ] >= -5
        )
    ][
        "StockCode"
    ]
)


selected = (
    eligible[
        eligible["StockCode"]
        .isin(
            reliable_codes
        )
    ]
    .sort_values(
        "Revenue",
        ascending=False
    )
    .head(10)
)


# ============================================================
# RUN SCENARIOS
# ============================================================

all_scenarios = []

for stock_code in selected[
    "StockCode"
]:

    result = generate_scenarios(
        stock_code
    )

    if result is not None:

        all_scenarios.append(
            result
        )


if all_scenarios:

    scenario_df = pd.concat(
        all_scenarios,
        ignore_index=True
    )

else:

    scenario_df = pd.DataFrame()


# ============================================================
# SCENARIO RESULTS
# ============================================================

print("\n" + "=" * 110)
print("PRICE SCENARIO ENGINE")
print("=" * 110)

if not scenario_df.empty:

    for stock_code in (
        scenario_df[
            "StockCode"
        ].unique()
    ):

        product_data = (
            scenario_df[
                scenario_df[
                    "StockCode"
                ]
                == stock_code
            ]
        )

        name = product_data.iloc[0][
            "Description"
        ]

        print(
            f"\n{name}"
        )

        print(
            "-" * 90
        )

        print(
            product_data[
                [
                    "Price_Change_Pct",
                    "Current_Price",
                    "New_Price",
                    "Final_Demand",
                    "Predicted_Revenue",
                    "Demand_Change_Pct",
                    "Revenue_Change_Pct"
                ]
            ]
            .round(2)
            .to_string(
                index=False
            )
        )


# ============================================================
# REVENUE SCENARIO SUMMARY
# ============================================================

print("\n" + "=" * 110)
print("REVENUE SCENARIO SUMMARY")
print("=" * 110)

if not scenario_df.empty:

    for stock_code in (
        scenario_df[
            "StockCode"
        ].unique()
    ):

        product_data = (
            scenario_df[
                scenario_df[
                    "StockCode"
                ]
                == stock_code
            ]
        )

        best_row = (
            product_data
            .loc[
                product_data[
                    "Predicted_Revenue"
                ].idxmax()
            ]
        )

        print(
            f"\n{best_row['Description']}"
        )

        print(
            f"Elasticity: "
            f"{best_row['Elasticity']:.3f}"
        )

        print(
            f"Elasticity R²: "
            f"{best_row['Elasticity_R2']:.3f}"
        )

        print(
            f"Evidence: "
            f"{best_row['Elasticity_Status']}"
        )

        print(
            f"Current Price: "
            f"£{best_row['Current_Price']:.2f}"
        )

        print(
            f"Scenario Price: "
            f"£{best_row['New_Price']:.2f}"
        )

        print(
            f"Price Change: "
            f"{best_row['Price_Change_Pct']:.0f}%"
        )

        print(
            f"Estimated Demand: "
            f"{best_row['Final_Demand']:.0f}"
        )

        print(
            f"Estimated Revenue: "
            f"£{best_row['Predicted_Revenue']:,.2f}"
        )


# ============================================================
# SANITY CHECK
# ============================================================

print("\n" + "=" * 110)
print("PRICE-DEMAND SANITY CHECK")
print("=" * 110)

violations = 0
products_tested = 0

if not scenario_df.empty:

    for stock_code in (
        scenario_df[
            "StockCode"
        ].unique()
    ):

        product_data = (
            scenario_df[
                scenario_df[
                    "StockCode"
                ]
                == stock_code
            ]
            .sort_values(
                "Price_Change_Pct"
            )
        )

        demands = (
            product_data[
                "Final_Demand"
            ]
            .values
        )

        # For negative elasticity,
        # demand should not increase
        # as price increases.

        if np.any(
            np.diff(demands) > 1e-6
        ):

            violations += 1

        products_tested += 1


print(
    f"Products with "
    f"price-demand violations: "
    f"{violations}"
)

print(
    f"Total products tested: "
    f"{products_tested}"
)


# ============================================================
# FINAL DIAGNOSTICS
# ============================================================

print("\n" + "=" * 110)
print("FINAL DIAGNOSTICS")
print("=" * 110)

print(
    f"Total products analysed: "
    f"{len(elasticity_df):,}"
)

print(
    f"Eligible products: "
    f"{len(eligible):,}"
)

print(
    f"Products used in simulator: "
    f"{len(selected)}"
)

print("\nEvidence distribution:")

print(
    elasticity_df[
        "Elasticity_Status"
    ]
    .value_counts()
    .to_string()
)

if not elasticity_df.empty:

    print(
        f"\nMinimum elasticity: "
        f"{elasticity_df['Elasticity'].min():.3f}"
    )

    print(
        f"Maximum elasticity: "
        f"{elasticity_df['Elasticity'].max():.3f}"
    )

    print(
        f"Median elasticity: "
        f"{elasticity_df['Elasticity'].median():.3f}"
    )

    print(
        f"Minimum elasticity R²: "
        f"{elasticity_df['Elasticity_R2'].min():.3f}"
    )

    print(
        f"Maximum elasticity R²: "
        f"{elasticity_df['Elasticity_R2'].max():.3f}"
    )


print(
    "\nScenario engine finished."
)