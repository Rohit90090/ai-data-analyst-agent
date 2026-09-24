from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MIN_OBSERVATIONS = 6
MIN_PRICE_POINTS = 3
MIN_PRICE_CHANGE = 0.01

RECENT_MONTHS = 3

PRICE_SCENARIOS = [-20, -10, 0, 10, 20]

# Non-product / administrative items
EXCLUDED_CODES = {
    "POSTAGE",
    "DOTCOMPOSTAGE",
    "M",
    "D",
    "S",
    "AMAZONFEE",
    "BANK CHARGES",
    "C2",
    "CRUK",
    "DCGS0003",
    "DCGS0004",
    "DCGS0005",
    "DCGS0006",
    "DCGS0007",
    "DCGS0008",
    "DCGS0009",
    "DCGS0016",
    "DCGS0017",
    "DCGS0019",
    "DCGS0020",
    "DCGS0021",
    "DCGS0022",
    "DCGS0024",
    "DCGS0025",
    "DCGS0026",
    "DCGS0030",
    "DCGS0031",
    "DCGS0032",
    "DCGS0033",
    "DCGS0034",
    "DCGS0035",
    "DCGS0036",
    "DCGS0037",
    "DCGS0038",
    "DCGS0040",
    "DCGS0041",
    "DCGS0042",
    "DCGS0043",
    "DCGS0044",
    "DCGS0045",
    "DCGS0046",
    "DCGS0047",
    "DCGS0048",
    "DCGS0049",
    "DCGS0050",
}


# ============================================================
# BASIC CLEANING
# ============================================================

def _prepare_data(df):
    """
    Standardize monthly demand data.
    """

    data = df.copy()

    required = [
        "StockCode",
        "Month",
        "Quantity",
        "Revenue",
        "Avg_Price",
    ]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Simulator data missing columns: {missing}"
        )

    data["StockCode"] = (
        data["StockCode"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    data["Month"] = pd.to_datetime(
        data["Month"],
        errors="coerce"
    )

    data["Quantity"] = pd.to_numeric(
        data["Quantity"],
        errors="coerce"
    )

    data["Revenue"] = pd.to_numeric(
        data["Revenue"],
        errors="coerce"
    )

    data["Avg_Price"] = pd.to_numeric(
        data["Avg_Price"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "StockCode",
            "Month",
            "Quantity",
            "Avg_Price",
        ]
    )

    data = data[
        (data["Quantity"] > 0)
        &
        (data["Avg_Price"] > 0)
    ].copy()

    # Remove administrative/non-product items
    data = data[
        ~data["StockCode"].isin(
            EXCLUDED_CODES
        )
    ].copy()

    # Description cleaning if available
    if "Description" in data.columns:

        data["Description"] = (
            data["Description"]
            .fillna("Unknown Product")
            .astype(str)
            .str.strip()
        )

        upper_desc = (
            data["Description"]
            .str.upper()
        )

        administrative_words = (
            upper_desc.str.contains(
                r"POSTAGE|BANK CHARGE|AMAZON FEE|MANUAL",
                regex=True,
                na=False,
            )
        )

        data = data[
            ~administrative_words
        ].copy()

    return data.sort_values(
        ["StockCode", "Month"]
    ).reset_index(drop=True)


# ============================================================
# ELASTICITY CALCULATION
# ============================================================

def calculate_elasticity(
    product_df,
    min_observations=MIN_OBSERVATIONS,
):
    """
    Estimate price elasticity using:

        Δlog(Quantity) = elasticity × Δlog(Price)

    This avoids fitting a highly over-parameterized
    product-specific model with month dummy variables.
    """

    data = product_df.copy()

    data = data.sort_values(
        "Month"
    )

    data["Log_Q"] = np.log(
        data["Quantity"].clip(lower=1e-6)
    )

    data["Log_P"] = np.log(
        data["Avg_Price"].clip(lower=1e-6)
    )

    data["Delta_Log_Q"] = (
        data["Log_Q"].diff()
    )

    data["Delta_Log_P"] = (
        data["Log_P"].diff()
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna(
        subset=[
            "Delta_Log_Q",
            "Delta_Log_P",
        ]
    )

    # Need enough observations
    if len(data) < min_observations - 1:

        return {
            "elasticity": np.nan,
            "r2": np.nan,
            "observations": len(data) + 1,
            "price_points": 0,
            "price_range": 0.0,
            "status": "Insufficient Data",
        }

    # Ignore months with effectively no price movement
    variation = (
        data["Delta_Log_P"]
        .abs()
        > MIN_PRICE_CHANGE
    )

    usable = data[
        variation
    ].copy()

    if len(usable) < 3:

        return {
            "elasticity": np.nan,
            "r2": np.nan,
            "observations": len(data) + 1,
            "price_points": 1,
            "price_range": 0.0,
            "status": "Insufficient Price Variation",
        }

    x = usable[
        "Delta_Log_P"
    ].to_numpy()

    y = usable[
        "Delta_Log_Q"
    ].to_numpy()

    # OLS slope through intercept
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    denominator = np.sum(
        (x - x_mean) ** 2
    )

    if denominator <= 1e-12:

        return {
            "elasticity": np.nan,
            "r2": np.nan,
            "observations": len(data) + 1,
            "price_points": len(usable),
            "price_range": 0.0,
            "status": "Insufficient Price Variation",
        }

    elasticity = (
        np.sum(
            (x - x_mean)
            *
            (y - y_mean)
        )
        /
        denominator
    )

    intercept = (
        y_mean
        -
        elasticity * x_mean
    )

    predictions = (
        intercept
        +
        elasticity * x
    )

    ss_res = np.sum(
        (y - predictions) ** 2
    )

    ss_tot = np.sum(
        (y - y_mean) ** 2
    )

    if ss_tot <= 1e-12:

        r2 = 0.0

    else:

        r2 = 1 - (
            ss_res / ss_tot
        )

    prices = (
        product_df["Avg_Price"]
        .dropna()
        .astype(float)
    )

    min_price = prices.min()
    max_price = prices.max()

    if min_price > 0:

        price_range = (
            max_price - min_price
        ) / min_price

    else:

        price_range = 0.0

    # --------------------------------------------------------
    # Evidence classification
    # --------------------------------------------------------

    abs_elasticity = abs(
        elasticity
    )

    if (
        elasticity < 0
        and
        abs_elasticity <= 8
        and
        r2 >= 0.15
        and
        price_range >= 0.10
        and
        len(usable) >= 5
    ):

        status = "Usable Evidence"

    elif (
        elasticity < 0
        and
        abs_elasticity <= 15
        and
        r2 >= 0.05
    ):

        status = "Moderate"

    elif (
        elasticity < 0
        and
        abs_elasticity <= 30
    ):

        status = "Weak Relationship"

    elif elasticity >= 0:

        status = "Unreliable Price Response"

    else:

        status = "High Sensitivity"

    return {
        "elasticity": float(
            elasticity
        ),
        "r2": float(
            max(0.0, min(1.0, r2))
        ),
        "observations": int(
            len(data) + 1
        ),
        "price_points": int(
            len(usable)
        ),
        "price_range": float(
            price_range
        ),
        "status": status,
    }


# ============================================================
# ALL PRODUCT ELASTICITIES
# ============================================================

def get_all_elasticities(
    monthly_df
):

    data = _prepare_data(
        monthly_df
    )

    results = []

    grouped = data.groupby(
        "StockCode",
        sort=False
    )

    for stock_code, product in grouped:

        result = calculate_elasticity(
            product
        )

        result["StockCode"] = (
            stock_code
        )

        if "Description" in product.columns:

            description = (
                product["Description"]
                .dropna()
                .astype(str)
            )

            if len(description) > 0:

                result["Description"] = (
                    description.iloc[-1]
                )

            else:

                result["Description"] = (
                    stock_code
                )

        else:

            result["Description"] = (
                stock_code
            )

        results.append(
            result
        )

    if not results:

        return pd.DataFrame(
            columns=[
                "StockCode",
                "Description",
                "elasticity",
                "r2",
                "observations",
                "price_points",
                "price_range",
                "status",
            ]
        )

    result_df = pd.DataFrame(
        results
    )

    # Product-level sorting
    result_df = result_df[
        [
            "StockCode",
            "Description",
            "elasticity",
            "r2",
            "observations",
            "price_points",
            "price_range",
            "status",
        ]
    ]

    return result_df.sort_values(
        "elasticity",
        na_position="last"
    ).reset_index(
        drop=True
    )


# ============================================================
# SELECT PRODUCT
# ============================================================

def get_product_data(
    monthly_df,
    stock_code
):

    data = _prepare_data(
        monthly_df
    )

    stock_code = str(
        stock_code
    ).strip().upper()

    product = data[
        data["StockCode"]
        == stock_code
    ].copy()

    if product.empty:

        raise ValueError(
            f"Product not found: {stock_code}"
        )

    return product.sort_values(
        "Month"
    ).reset_index(
        drop=True
    )


# ============================================================
# RECENT BASELINE
# ============================================================

def _get_recent_baseline(
    product_df
):

    product = product_df.sort_values(
        "Month"
    ).copy()

    recent = product.tail(
        RECENT_MONTHS
    )

    if recent.empty:

        raise ValueError(
            "No recent observations available."
        )

    current_price = float(
        recent["Avg_Price"].median()
    )

    recent_demand = float(
        recent["Quantity"].mean()
    )

    recent_revenue = float(
        recent["Revenue"].mean()
    )

    return {
        "current_price": current_price,
        "current_demand": recent_demand,
        "current_revenue": recent_revenue,
    }


# ============================================================
# SAFE ELASTICITY
# ============================================================

def _safe_elasticity(
    elasticity_result
):

    if elasticity_result is None:

        return None

    elasticity = (
        elasticity_result
        .get("elasticity")
    )

    if elasticity is None:
        return None

    if not np.isfinite(
        elasticity
    ):

        return None

    # Positive elasticity is not suitable
    # for a normal price-demand simulator.
    if elasticity >= 0:

        return None

    # Extremely large estimates are not trusted.
    if abs(elasticity) > 30:

        return None

    return float(
        elasticity
    )


# ============================================================
# SCENARIO GENERATION
# ============================================================

def generate_scenarios(
    product_df,
    elasticity_result,
):

    product = _prepare_data(
        product_df
    )

    if product.empty:

        raise ValueError(
            "No usable product data."
        )

    baseline = _get_recent_baseline(
        product
    )

    current_price = baseline[
        "current_price"
    ]

    current_demand = baseline[
        "current_demand"
    ]

    current_revenue = (
        current_price
        *
        current_demand
    )

    elasticity = _safe_elasticity(
        elasticity_result
    )

    r2 = np.nan

    status = "Insufficient Evidence"

    if elasticity_result:

        r2 = elasticity_result.get(
            "r2",
            np.nan
        )

        status = elasticity_result.get(
            "status",
            "Unknown"
        )

    rows = []

    for change in PRICE_SCENARIOS:

        price_multiplier = (
            1
            +
            change / 100
        )

        scenario_price = (
            current_price
            *
            price_multiplier
        )

        # ----------------------------------------------------
        # Demand calculation
        # ----------------------------------------------------

        if (
            elasticity is not None
            and
            scenario_price > 0
        ):

            # Constant-elasticity demand model:
            #
            # Q1 / Q0 = (P1 / P0)^elasticity
            #
            demand_multiplier = (
                price_multiplier
                **
                elasticity
            )

            estimated_demand = (
                current_demand
                *
                demand_multiplier
            )

        else:

            # No reliable elasticity:
            # keep demand at baseline and clearly
            # mark the evidence as insufficient.
            estimated_demand = (
                current_demand
            )

        # ----------------------------------------------------
        # Revenue
        # ----------------------------------------------------

        estimated_revenue = (
            scenario_price
            *
            estimated_demand
        )

        revenue_change = (
            (
                estimated_revenue
                /
                current_revenue
            )
            - 1
        ) * 100

        demand_change = (
            (
                estimated_demand
                /
                current_demand
            )
            - 1
        ) * 100

        rows.append(
            {
                "Price_Change_Pct": change,
                "Scenario_Price": scenario_price,
                "Estimated_Demand": estimated_demand,
                "Estimated_Revenue": estimated_revenue,
                "Demand_Change_Pct": demand_change,
                "Revenue_Change_Pct": revenue_change,
                "Elasticity": (
                    elasticity
                    if elasticity is not None
                    else np.nan
                ),
                "R2": r2,
                "Evidence_Status": status,
            }
        )

    result = pd.DataFrame(
        rows
    )

    return result


# ============================================================
# PRODUCT SCENARIO SUMMARY
# ============================================================

def build_product_scenario_summary(
    monthly_df,
    stock_code
):

    product = get_product_data(
        monthly_df,
        stock_code
    )

    elasticity_table = (
        get_all_elasticities(
            product
        )
    )

    if elasticity_table.empty:

        elasticity_result = {
            "elasticity": np.nan,
            "r2": np.nan,
            "status": "Insufficient Evidence",
        }

    else:

        elasticity_result = (
            elasticity_table.iloc[0]
            .to_dict()
        )

    scenarios = generate_scenarios(
        product,
        elasticity_result
    )

    return scenarios


# ============================================================
# FIND PRODUCTS SUITABLE FOR SIMULATOR
# ============================================================

def get_simulator_products(
    monthly_df,
    limit=100
):

    data = _prepare_data(
        monthly_df
    )

    results = []

    grouped = data.groupby(
        "StockCode",
        sort=False
    )

    for stock_code, product in grouped:

        if len(product) < MIN_OBSERVATIONS:

            continue

        prices = (
            product["Avg_Price"]
            .dropna()
            .unique()
        )

        if len(prices) < MIN_PRICE_POINTS:

            continue

        elasticity_result = (
            calculate_elasticity(
                product
            )
        )

        elasticity = (
            elasticity_result
            .get("elasticity")
        )

        if elasticity is None:
            continue

        if not np.isfinite(
            elasticity
        ):
            continue

        # Only keep negative elasticity
        # because the simulator models normal
        # price-demand behavior.
        if elasticity >= 0:
            continue

        if abs(elasticity) > 30:
            continue

        description = stock_code

        if "Description" in product.columns:

            descriptions = (
                product["Description"]
                .dropna()
                .astype(str)
            )

            if not descriptions.empty:

                description = (
                    descriptions.iloc[-1]
                )

        results.append(
            {
                "StockCode": stock_code,
                "Description": description,
                "Elasticity": elasticity,
                "R2": elasticity_result.get(
                    "r2",
                    np.nan
                ),
                "Observations": elasticity_result.get(
                    "observations",
                    len(product)
                ),
                "Price_Points": elasticity_result.get(
                    "price_points",
                    len(prices)
                ),
                "Evidence_Status": elasticity_result.get(
                    "status",
                    "Unknown"
                ),
            }
        )

    result = pd.DataFrame(
        results
    )

    if result.empty:

        return result

    # Prefer stronger evidence
    result["Evidence_Rank"] = (
        result["R2"]
        .fillna(0)
    )

    result = result.sort_values(
        [
            "Evidence_Rank",
            "Observations",
        ],
        ascending=False
    )

    return result.head(
        limit
    ).drop(
        columns=["Evidence_Rank"]
    ).reset_index(
        drop=True
    )