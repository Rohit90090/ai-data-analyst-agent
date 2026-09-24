import pandas as pd
import numpy as np


# =========================================================
# BASIC HELPERS
# =========================================================

def money(value):
    """Format a number as currency."""
    try:
        value = float(value)

        if abs(value) >= 1_000_000:
            return f"£{value / 1_000_000:.2f}M"

        if abs(value) >= 1_000:
            return f"£{value / 1_000:.2f}K"

        return f"£{value:,.2f}"

    except Exception:
        return "N/A"


def number(value):
    """Format numeric values."""
    try:
        value = float(value)

        if value.is_integer():
            return f"{int(value):,}"

        return f"{value:,.2f}"

    except Exception:
        return "N/A"


def percentage(value):
    """Format percentage."""
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "N/A"


# =========================================================
# COLUMN DETECTION
# =========================================================

def detect_columns(df):
    """
    Automatically detects important business columns
    from an uploaded dataset.
    """

    columns = list(df.columns)

    lower_map = {
        str(col).lower().strip(): col
        for col in columns
    }

    def find_column(keywords):
        for col in columns:
            name = str(col).lower().strip()

            for keyword in keywords:
                if keyword in name:
                    return col

        return None

    revenue_col = find_column([
        "revenue",
        "sales",
        "sales_amount",
        "total_sales",
        "amount",
        "turnover",
        "income"
    ])

    quantity_col = find_column([
        "quantity",
        "qty",
        "units",
        "sold"
    ])

    product_col = find_column([
        "product",
        "description",
        "item",
        "sku",
        "stockcode"
    ])

    customer_col = find_column([
        "customer",
        "customer_id",
        "client",
        "user"
    ])

    country_col = find_column([
        "country",
        "region",
        "location",
        "state",
        "city"
    ])

    date_col = None

    for col in columns:

        name = str(col).lower().strip()

        if any(keyword in name for keyword in [
            "date",
            "month",
            "year",
            "time"
        ]):

            converted = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            if converted.notna().sum() > 0:
                date_col = col
                break

    transaction_col = find_column([
        "transaction",
        "invoice",
        "order",
        "order_id",
        "invoice_id"
    ])

    return {
        "revenue": revenue_col,
        "quantity": quantity_col,
        "product": product_col,
        "customer": customer_col,
        "country": country_col,
        "date": date_col,
        "transaction": transaction_col
    }


# =========================================================
# DATASET OVERVIEW
# =========================================================

def generate_overview(df):

    rows = len(df)

    columns = len(df.columns)

    missing = int(
        df.isna().sum().sum()
    )

    duplicates = int(
        df.duplicated().sum()
    )

    missing_percentage = (
        missing / (rows * columns) * 100
        if rows > 0 and columns > 0
        else 0
    )

    return {
        "rows": rows,
        "columns": columns,
        "missing": missing,
        "missing_percentage": missing_percentage,
        "duplicates": duplicates
    }


# =========================================================
# REVENUE ANALYSIS
# =========================================================

def generate_revenue_analysis(df, revenue_col):

    if revenue_col is None:
        return None

    series = pd.to_numeric(
        df[revenue_col],
        errors="coerce"
    ).dropna()

    if series.empty:
        return None

    total = series.sum()

    average = series.mean()

    maximum = series.max()

    minimum = series.min()

    return {
        "total": total,
        "average": average,
        "maximum": maximum,
        "minimum": minimum
    }


# =========================================================
# QUANTITY ANALYSIS
# =========================================================

def generate_quantity_analysis(df, quantity_col):

    if quantity_col is None:
        return None

    series = pd.to_numeric(
        df[quantity_col],
        errors="coerce"
    ).dropna()

    if series.empty:
        return None

    return {
        "total": series.sum(),
        "average": series.mean(),
        "maximum": series.max(),
        "minimum": series.min()
    }


# =========================================================
# PRODUCT ANALYSIS
# =========================================================

def generate_product_analysis(
    df,
    product_col,
    revenue_col,
    quantity_col
):

    if product_col is None:
        return None

    working = df.copy()

    working[product_col] = (
        working[product_col]
        .astype(str)
        .str.strip()
    )

    working = working[
        working[product_col].notna()
    ]

    if working.empty:
        return None

    result = {}

    # -----------------------------
    # TOP PRODUCTS BY REVENUE
    # -----------------------------

    if revenue_col:

        working[revenue_col] = pd.to_numeric(
            working[revenue_col],
            errors="coerce"
        )

        top_revenue = (
            working
            .groupby(product_col)[revenue_col]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        top_revenue.columns = [
            "Product",
            "Revenue"
        ]

        result["top_revenue"] = top_revenue

    # -----------------------------
    # TOP PRODUCTS BY QUANTITY
    # -----------------------------

    if quantity_col:

        working[quantity_col] = pd.to_numeric(
            working[quantity_col],
            errors="coerce"
        )

        top_quantity = (
            working
            .groupby(product_col)[quantity_col]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        top_quantity.columns = [
            "Product",
            "Quantity"
        ]

        result["top_quantity"] = top_quantity

    return result


# =========================================================
# CUSTOMER ANALYSIS
# =========================================================

def generate_customer_analysis(
    df,
    customer_col,
    revenue_col
):

    if customer_col is None:
        return None

    customers = df[customer_col].nunique(
        dropna=True
    )

    result = {
        "unique_customers": customers
    }

    if revenue_col:

        temp = df[
            [customer_col, revenue_col]
        ].copy()

        temp[revenue_col] = pd.to_numeric(
            temp[revenue_col],
            errors="coerce"
        )

        customer_revenue = (
            temp
            .groupby(customer_col)[revenue_col]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        customer_revenue.columns = [
            "Customer",
            "Revenue"
        ]

        result["top_customers"] = customer_revenue

    return result


# =========================================================
# COUNTRY / REGION ANALYSIS
# =========================================================

def generate_country_analysis(
    df,
    country_col,
    revenue_col
):

    if country_col is None:
        return None

    result = {}

    if revenue_col:

        temp = df[
            [country_col, revenue_col]
        ].copy()

        temp[revenue_col] = pd.to_numeric(
            temp[revenue_col],
            errors="coerce"
        )

        top_countries = (
            temp
            .groupby(country_col)[revenue_col]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        top_countries.columns = [
            "Country",
            "Revenue"
        ]

        result["top_countries"] = top_countries

    return result


# =========================================================
# DATE / TREND ANALYSIS
# =========================================================

def generate_trend_analysis(
    df,
    date_col,
    revenue_col
):

    if date_col is None or revenue_col is None:
        return None

    temp = df[
        [date_col, revenue_col]
    ].copy()

    temp[date_col] = pd.to_datetime(
        temp[date_col],
        errors="coerce"
    )

    temp[revenue_col] = pd.to_numeric(
        temp[revenue_col],
        errors="coerce"
    )

    temp = temp.dropna()

    if temp.empty:
        return None

    temp["Period"] = (
        temp[date_col]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        temp
        .groupby("Period")[revenue_col]
        .sum()
        .reset_index()
    )

    monthly.columns = [
        "Month",
        "Revenue"
    ]

    monthly["Revenue"] = monthly["Revenue"].round(2)

    result = {
        "monthly": monthly
    }

    if not monthly.empty:

        highest = monthly.loc[
            monthly["Revenue"].idxmax()
        ]

        lowest = monthly.loc[
            monthly["Revenue"].idxmin()
        ]

        result["highest_month"] = highest["Month"]

        result["highest_month_revenue"] = highest["Revenue"]

        result["lowest_month"] = lowest["Month"]

        result["lowest_month_revenue"] = lowest["Revenue"]

    return result


# =========================================================
# DATA QUALITY ANALYSIS
# =========================================================

def generate_quality_analysis(df):

    missing_by_column = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing_by_column = (
        missing_by_column[
            missing_by_column > 0
        ]
        .reset_index()
    )

    missing_by_column.columns = [
        "Column",
        "Missing Values"
    ]

    return {
        "missing_by_column": missing_by_column,
        "duplicate_rows": int(
            df.duplicated().sum()
        )
    }


# =========================================================
# AUTOMATIC BUSINESS INSIGHTS
# =========================================================

def generate_business_insights(
    df,
    columns,
    revenue_analysis,
    quantity_analysis,
    product_analysis,
    customer_analysis,
    country_analysis,
    trend_analysis,
    quality_analysis
):

    insights = []

    # -----------------------------
    # REVENUE INSIGHT
    # -----------------------------

    if revenue_analysis:

        insights.append(
            f"Total revenue generated is "
            f"{money(revenue_analysis['total'])}."
        )

        insights.append(
            f"Average revenue per record is "
            f"{money(revenue_analysis['average'])}."
        )

    # -----------------------------
    # TOP PRODUCT
    # -----------------------------

    if product_analysis:

        top_revenue = product_analysis.get(
            "top_revenue"
        )

        if (
            isinstance(top_revenue, pd.DataFrame)
            and not top_revenue.empty
        ):

            product = top_revenue.iloc[0]["Product"]

            revenue = top_revenue.iloc[0]["Revenue"]

            insights.append(
                f"The highest-revenue product is "
                f"'{product}', generating "
                f"{money(revenue)}."
            )

    # -----------------------------
    # QUANTITY
    # -----------------------------

    if quantity_analysis:

        insights.append(
            f"Total quantity recorded is "
            f"{number(quantity_analysis['total'])}."
        )

    # -----------------------------
    # CUSTOMER
    # -----------------------------

    if customer_analysis:

        customers = customer_analysis[
            "unique_customers"
        ]

        insights.append(
            f"The dataset contains "
            f"{number(customers)} unique customers."
        )

    # -----------------------------
    # COUNTRY
    # -----------------------------

    if country_analysis:

        top_countries = country_analysis.get(
            "top_countries"
        )

        if (
            isinstance(top_countries, pd.DataFrame)
            and not top_countries.empty
        ):

            country = top_countries.iloc[0]["Country"]

            revenue = top_countries.iloc[0]["Revenue"]

            insights.append(
                f"The highest-revenue country/region is "
                f"'{country}', with "
                f"{money(revenue)} in revenue."
            )

    # -----------------------------
    # TREND
    # -----------------------------

    if trend_analysis:

        highest_month = trend_analysis.get(
            "highest_month"
        )

        highest_revenue = trend_analysis.get(
            "highest_month_revenue"
        )

        if highest_month:

            insights.append(
                f"The highest-revenue period is "
                f"{highest_month}, generating "
                f"{money(highest_revenue)}."
            )

    # -----------------------------
    # DATA QUALITY
    # -----------------------------

    if quality_analysis:

        duplicate_rows = quality_analysis[
            "duplicate_rows"
        ]

        missing_table = quality_analysis[
            "missing_by_column"
        ]

        if duplicate_rows > 0:

            insights.append(
                f"The dataset contains "
                f"{number(duplicate_rows)} duplicate rows "
                f"that may require review."
            )

        if (
            isinstance(missing_table, pd.DataFrame)
            and not missing_table.empty
        ):

            column = missing_table.iloc[0]["Column"]

            missing = missing_table.iloc[0][
                "Missing Values"
            ]

            insights.append(
                f"The column '{column}' contains "
                f"{number(missing)} missing values."
            )

    if not insights:

        insights.append(
            "No major automatic business insight "
            "could be generated from the available columns."
        )

    return insights


# =========================================================
# MAIN REPORT GENERATOR
# =========================================================

def generate_report(df):

    if df is None:
        return {
            "success": False,
            "message": "No dataset available."
        }

    if not isinstance(df, pd.DataFrame):
        return {
            "success": False,
            "message": "Invalid dataset format."
        }

    if df.empty:
        return {
            "success": False,
            "message": "The uploaded dataset is empty."
        }

    # -----------------------------
    # Detect columns
    # -----------------------------

    columns = detect_columns(df)

    # -----------------------------
    # Generate sections
    # -----------------------------

    overview = generate_overview(df)

    revenue_analysis = generate_revenue_analysis(
        df,
        columns["revenue"]
    )

    quantity_analysis = generate_quantity_analysis(
        df,
        columns["quantity"]
    )

    product_analysis = generate_product_analysis(
        df,
        columns["product"],
        columns["revenue"],
        columns["quantity"]
    )

    customer_analysis = generate_customer_analysis(
        df,
        columns["customer"],
        columns["revenue"]
    )

    country_analysis = generate_country_analysis(
        df,
        columns["country"],
        columns["revenue"]
    )

    trend_analysis = generate_trend_analysis(
        df,
        columns["date"],
        columns["revenue"]
    )

    quality_analysis = generate_quality_analysis(
        df
    )

    # -----------------------------
    # Generate insights
    # -----------------------------

    insights = generate_business_insights(
        df,
        columns,
        revenue_analysis,
        quantity_analysis,
        product_analysis,
        customer_analysis,
        country_analysis,
        trend_analysis,
        quality_analysis
    )

    # -----------------------------
    # Final report
    # -----------------------------

    return {
        "success": True,

        "title": "AI Analyst Report",

        "dataset": {
            "rows": overview["rows"],
            "columns": overview["columns"]
        },

        "overview": overview,

        "detected_columns": columns,

        "revenue": revenue_analysis,

        "quantity": quantity_analysis,

        "products": product_analysis,

        "customers": customer_analysis,

        "countries": country_analysis,

        "trend": trend_analysis,

        "data_quality": quality_analysis,

        "insights": insights
    }