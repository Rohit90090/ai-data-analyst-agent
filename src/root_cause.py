import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# COLUMN DETECTION
# ---------------------------------------------------------

def find_column(df, keywords):
    columns = list(df.columns)

    for keyword in keywords:
        for col in columns:
            if keyword.lower() == str(col).lower():
                return col

    for keyword in keywords:
        for col in columns:
            if keyword.lower() in str(col).lower():
                return col

    return None


def find_date_column(df):
    return find_column(
        df,
        [
            "invoicedate",
            "invoice_date",
            "date",
            "datetime",
            "month"
        ]
    )


def find_revenue_column(df):
    return find_column(
        df,
        [
            "revenue",
            "sales",
            "sales_amount",
            "amount",
            "total_revenue"
        ]
    )


def find_quantity_column(df):
    return find_column(
        df,
        [
            "quantity",
            "qty",
            "units",
            "units_sold"
        ]
    )


def find_product_column(df):
    return find_column(
        df,
        [
            "description",
            "product",
            "product_name",
            "item"
        ]
    )


def find_country_column(df):
    return find_column(
        df,
        [
            "country",
            "region",
            "market"
        ]
    )


def find_transaction_column(df):
    return find_column(
        df,
        [
            "invoice",
            "transaction",
            "transaction_id",
            "order_id"
        ]
    )


def find_customer_column(df):
    return find_column(
        df,
        [
            "customer_id",
            "customerid",
            "customer"
        ]
    )


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def clean_number(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)


def pct_change(old, new):
    if old == 0:
        return np.nan
    return ((new - old) / abs(old)) * 100


def money(value):
    if pd.isna(value):
        return "N/A"
    return f"£{value:,.2f}"


# ---------------------------------------------------------
# DATE PREPARATION
# ---------------------------------------------------------

def prepare_dates(df):
    df = df.copy()

    date_col = find_date_column(df)

    if date_col is None:
        return df, None

    df["_rc_date"] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df = df.dropna(subset=["_rc_date"])

    return df, "_rc_date"


# ---------------------------------------------------------
# MONTHLY METRICS
# ---------------------------------------------------------

def monthly_metrics(df):
    df, date_col = prepare_dates(df)

    if date_col is None or df.empty:
        return pd.DataFrame()

    revenue_col = find_revenue_column(df)
    quantity_col = find_quantity_column(df)
    transaction_col = find_transaction_column(df)

    if revenue_col is None:
        return pd.DataFrame()

    df["_rc_revenue"] = clean_number(df[revenue_col])

    if quantity_col:
        df["_rc_quantity"] = clean_number(df[quantity_col])
    else:
        df["_rc_quantity"] = 0

    df["_rc_month"] = df["_rc_date"].dt.to_period("M").astype(str)

    result = (
        df.groupby("_rc_month")
        .agg(
            Revenue=("_rc_revenue", "sum"),
            Quantity=("_rc_quantity", "sum")
        )
        .reset_index()
    )

    if transaction_col:
        transaction_counts = (
            df.groupby("_rc_month")[transaction_col]
            .nunique()
            .reset_index(name="Transactions")
        )

        result = result.merge(
            transaction_counts,
            on="_rc_month",
            how="left"
        )
    else:
        result["Transactions"] = 0

    result["AOV"] = np.where(
        result["Transactions"] > 0,
        result["Revenue"] / result["Transactions"],
        0
    )

    return result


# ---------------------------------------------------------
# PRODUCT CHANGE ANALYSIS
# ---------------------------------------------------------

def product_change_analysis(df, previous_month, current_month):
    df, date_col = prepare_dates(df)

    if date_col is None:
        return pd.DataFrame()

    product_col = find_product_column(df)
    revenue_col = find_revenue_column(df)
    quantity_col = find_quantity_column(df)

    if product_col is None or revenue_col is None:
        return pd.DataFrame()

    df["_rc_revenue"] = clean_number(df[revenue_col])

    if quantity_col:
        df["_rc_quantity"] = clean_number(df[quantity_col])
    else:
        df["_rc_quantity"] = 0

    df["_rc_month"] = df["_rc_date"].dt.to_period("M").astype(str)

    filtered = df[
        df["_rc_month"].isin(
            [previous_month, current_month]
        )
    ].copy()

    if filtered.empty:
        return pd.DataFrame()

    pivot = (
        filtered.groupby(
            ["_rc_month", product_col]
        )
        .agg(
            Revenue=("_rc_revenue", "sum"),
            Quantity=("_rc_quantity", "sum")
        )
        .reset_index()
    )

    revenue_pivot = pivot.pivot_table(
        index=product_col,
        columns="_rc_month",
        values="Revenue",
        aggfunc="sum",
        fill_value=0
    )

    quantity_pivot = pivot.pivot_table(
        index=product_col,
        columns="_rc_month",
        values="Quantity",
        aggfunc="sum",
        fill_value=0
    )

    if previous_month not in revenue_pivot.columns:
        revenue_pivot[previous_month] = 0

    if current_month not in revenue_pivot.columns:
        revenue_pivot[current_month] = 0

    if previous_month not in quantity_pivot.columns:
        quantity_pivot[previous_month] = 0

    if current_month not in quantity_pivot.columns:
        quantity_pivot[current_month] = 0

    result = pd.DataFrame(index=revenue_pivot.index)

    result["Previous Revenue"] = revenue_pivot[previous_month]
    result["Current Revenue"] = revenue_pivot[current_month]

    result["Revenue Change"] = (
        result["Current Revenue"]
        - result["Previous Revenue"]
    )

    result["Revenue Change %"] = np.where(
        result["Previous Revenue"] != 0,
        result["Revenue Change"]
        / result["Previous Revenue"] * 100,
        np.nan
    )

    result["Previous Quantity"] = quantity_pivot[previous_month]
    result["Current Quantity"] = quantity_pivot[current_month]

    result["Quantity Change"] = (
        result["Current Quantity"]
        - result["Previous Quantity"]
    )

    result["Quantity Change %"] = np.where(
        result["Previous Quantity"] != 0,
        result["Quantity Change"]
        / result["Previous Quantity"] * 100,
        np.nan
    )

    result = result.reset_index()

    result = result.sort_values(
        "Revenue Change"
    )

    return result


# ---------------------------------------------------------
# COUNTRY ANALYSIS
# ---------------------------------------------------------

def country_change_analysis(df, previous_month, current_month):
    df, date_col = prepare_dates(df)

    if date_col is None:
        return pd.DataFrame()

    country_col = find_country_column(df)
    revenue_col = find_revenue_column(df)

    if country_col is None or revenue_col is None:
        return pd.DataFrame()

    df["_rc_revenue"] = clean_number(df[revenue_col])
    df["_rc_month"] = df["_rc_date"].dt.to_period("M").astype(str)

    filtered = df[
        df["_rc_month"].isin(
            [previous_month, current_month]
        )
    ]

    result = (
        filtered.groupby(
            ["_rc_month", country_col]
        )["_rc_revenue"]
        .sum()
        .reset_index()
    )

    pivot = result.pivot_table(
        index=country_col,
        columns="_rc_month",
        values="_rc_revenue",
        aggfunc="sum",
        fill_value=0
    )

    if previous_month not in pivot.columns:
        pivot[previous_month] = 0

    if current_month not in pivot.columns:
        pivot[current_month] = 0

    pivot["Revenue Change"] = (
        pivot[current_month]
        - pivot[previous_month]
    )

    pivot["Revenue Change %"] = np.where(
        pivot[previous_month] != 0,
        pivot["Revenue Change"]
        / pivot[previous_month] * 100,
        np.nan
    )

    return pivot.reset_index().sort_values(
        "Revenue Change"
    )


# ---------------------------------------------------------
# MAIN ROOT-CAUSE ENGINE
# ---------------------------------------------------------

def analyze_root_cause(df, current_month=None):

    monthly = monthly_metrics(df)

    if monthly.empty:
        return {
            "success": False,
            "message": (
                "Date and revenue columns could not be detected."
            )
        }

    monthly = monthly.sort_values("_rc_month")

    available_months = monthly["_rc_month"].tolist()

    if len(available_months) < 2:
        return {
            "success": False,
            "message": "At least two months are required."
        }

    if current_month is None:
        current_month = available_months[-1]

    if current_month not in available_months:
        return {
            "success": False,
            "message": (
                f"Month {current_month} was not found."
            )
        }

    current_index = available_months.index(current_month)

    if current_index == 0:
        return {
            "success": False,
            "message": "No previous month is available."
        }

    previous_month = available_months[current_index - 1]

    previous = monthly[
        monthly["_rc_month"] == previous_month
    ].iloc[0]

    current = monthly[
        monthly["_rc_month"] == current_month
    ].iloc[0]

    revenue_change = current["Revenue"] - previous["Revenue"]
    revenue_change_pct = pct_change(
        previous["Revenue"],
        current["Revenue"]
    )

    quantity_change = current["Quantity"] - previous["Quantity"]
    quantity_change_pct = pct_change(
        previous["Quantity"],
        current["Quantity"]
    )

    transaction_change = (
        current["Transactions"]
        - previous["Transactions"]
    )

    transaction_change_pct = pct_change(
        previous["Transactions"],
        current["Transactions"]
    )

    aov_change = current["AOV"] - previous["AOV"]

    aov_change_pct = pct_change(
        previous["AOV"],
        current["AOV"]
    )

    # -----------------------------------------------------
    # DRIVER DETECTION
    # -----------------------------------------------------

    drivers = []

    if quantity_change_pct < -5:
        drivers.append({
            "Driver": "Quantity",
            "Impact": "Negative",
            "Change": quantity_change_pct,
            "Explanation": (
                "Units sold decreased significantly."
            )
        })

    elif quantity_change_pct > 5:
        drivers.append({
            "Driver": "Quantity",
            "Impact": "Positive",
            "Change": quantity_change_pct,
            "Explanation": (
                "Units sold increased."
            )
        })

    if transaction_change_pct < -5:
        drivers.append({
            "Driver": "Transactions",
            "Impact": "Negative",
            "Change": transaction_change_pct,
            "Explanation": (
                "The number of transactions decreased."
            )
        })

    elif transaction_change_pct > 5:
        drivers.append({
            "Driver": "Transactions",
            "Impact": "Positive",
            "Change": transaction_change_pct,
            "Explanation": (
                "The number of transactions increased."
            )
        })

    if aov_change_pct < -5:
        drivers.append({
            "Driver": "Average Order Value",
            "Impact": "Negative",
            "Change": aov_change_pct,
            "Explanation": (
                "Average revenue generated per transaction decreased."
            )
        })

    elif aov_change_pct > 5:
        drivers.append({
            "Driver": "Average Order Value",
            "Impact": "Positive",
            "Change": aov_change_pct,
            "Explanation": (
                "Average revenue generated per transaction increased."
            )
        })

    product_changes = product_change_analysis(
        df,
        previous_month,
        current_month
    )

    country_changes = country_change_analysis(
        df,
        previous_month,
        current_month
    )

    # -----------------------------------------------------
    # TOP NEGATIVE PRODUCTS
    # -----------------------------------------------------

    top_negative_products = pd.DataFrame()

    if not product_changes.empty:
        top_negative_products = (
            product_changes[
                product_changes["Revenue Change"] < 0
            ]
            .sort_values("Revenue Change")
            .head(10)
            .copy()
        )

    # -----------------------------------------------------
    # TOP POSITIVE PRODUCTS
    # -----------------------------------------------------

    top_positive_products = pd.DataFrame()

    if not product_changes.empty:
        top_positive_products = (
            product_changes[
                product_changes["Revenue Change"] > 0
            ]
            .sort_values(
                "Revenue Change",
                ascending=False
            )
            .head(10)
            .copy()
        )

    # -----------------------------------------------------
    # COUNTRY DRIVERS
    # -----------------------------------------------------

    top_negative_countries = pd.DataFrame()

    if not country_changes.empty:
        top_negative_countries = (
            country_changes[
                country_changes["Revenue Change"] < 0
            ]
            .sort_values("Revenue Change")
            .head(10)
            .copy()
        )

    # -----------------------------------------------------
    # EXPLANATION
    # -----------------------------------------------------

    explanations = []

    if revenue_change < 0:

        if quantity_change_pct < -5:
            explanations.append(
                f"Revenue decreased by {abs(revenue_change_pct):.1f}%, "
                f"while quantity decreased by "
                f"{abs(quantity_change_pct):.1f}%."
            )

        if transaction_change_pct < -5:
            explanations.append(
                f"Transactions decreased by "
                f"{abs(transaction_change_pct):.1f}%."
            )

        if aov_change_pct < -5:
            explanations.append(
                f"Average order value decreased by "
                f"{abs(aov_change_pct):.1f}%."
            )

        if not top_negative_products.empty:

            product_name = str(
                top_negative_products.iloc[0][
                    product_changes.columns[0]
                ]
            )

            product_loss = abs(
                top_negative_products.iloc[0][
                    "Revenue Change"
                ]
            )

            explanations.append(
                f"The largest product-level revenue decline "
                f"came from {product_name}, contributing "
                f"approximately {money(product_loss)} "
                f"of negative change."
            )

        if not top_negative_countries.empty:

            country_name = str(
                top_negative_countries.iloc[0][
                    country_changes.columns[0]
                ]
            )

            country_loss = abs(
                top_negative_countries.iloc[0][
                    "Revenue Change"
                ]
            )

            explanations.append(
                f"The largest country-level decline came from "
                f"{country_name}, with approximately "
                f"{money(country_loss)} lower revenue."
            )

    else:

        explanations.append(
            f"Revenue increased by {revenue_change_pct:.1f}% "
            f"from {previous_month} to {current_month}."
        )

        if quantity_change_pct > 5:
            explanations.append(
                f"Quantity increased by "
                f"{quantity_change_pct:.1f}%."
            )

        if transaction_change_pct > 5:
            explanations.append(
                f"Transactions increased by "
                f"{transaction_change_pct:.1f}%."
            )

        if aov_change_pct > 5:
            explanations.append(
                f"Average order value increased by "
                f"{aov_change_pct:.1f}%."
            )

    return {
        "success": True,

        "previous_month": previous_month,
        "current_month": current_month,

        "previous_revenue": previous["Revenue"],
        "current_revenue": current["Revenue"],
        "revenue_change": revenue_change,
        "revenue_change_pct": revenue_change_pct,

        "previous_quantity": previous["Quantity"],
        "current_quantity": current["Quantity"],
        "quantity_change": quantity_change,
        "quantity_change_pct": quantity_change_pct,

        "previous_transactions": previous["Transactions"],
        "current_transactions": current["Transactions"],
        "transaction_change": transaction_change,
        "transaction_change_pct": transaction_change_pct,

        "previous_aov": previous["AOV"],
        "current_aov": current["AOV"],
        "aov_change": aov_change,
        "aov_change_pct": aov_change_pct,

        "drivers": pd.DataFrame(drivers),
        "product_changes": product_changes,
        "country_changes": country_changes,
        "top_negative_products": top_negative_products,
        "top_positive_products": top_positive_products,
        "top_negative_countries": top_negative_countries,

        "explanations": explanations,

        "monthly": monthly
    }