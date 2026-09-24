import re
import numpy as np
import pandas as pd


# ============================================================
# COLUMN DETECTION
# ============================================================

def _normalized_columns(df):
    return [(col, str(col).lower().strip()) for col in df.columns]


def find_column(df, keywords):
    columns = _normalized_columns(df)

    # Exact match first
    for keyword in keywords:
        keyword = keyword.lower().strip()

        for original, normalized in columns:
            if normalized == keyword:
                return original

    # Partial match second
    for keyword in keywords:
        keyword = keyword.lower().strip()

        for original, normalized in columns:
            if keyword in normalized:
                return original

    return None


def find_product_column(df):
    preferred = [
        "description",
        "product_name",
        "product",
        "item_name",
        "item"
    ]

    for name in preferred:
        for col in df.columns:
            if str(col).lower().strip() == name:
                return col

    return find_column(df, preferred)


def find_revenue_column(df):
    preferred = [
        "revenue",
        "sales",
        "total_sales",
        "sales_amount",
        "amount",
        "value"
    ]

    for name in preferred:
        for col in df.columns:
            if str(col).lower().strip() == name:
                return col

    return find_column(df, preferred)


def find_quantity_column(df):
    preferred = [
        "quantity",
        "qty",
        "units_sold",
        "units",
        "quantity_sold"
    ]

    for name in preferred:
        for col in df.columns:
            if str(col).lower().strip() == name:
                return col

    return find_column(df, preferred)


def find_country_column(df):
    return find_column(
        df,
        [
            "country",
            "region",
            "market",
            "location"
        ]
    )


def find_month_column(df):
    preferred = [
        "month",
        "invoice_date",
        "order_date",
        "transaction_date",
        "date",
        "datetime",
        "timestamp"
    ]

    for name in preferred:
        for col in df.columns:
            if str(col).lower().strip() == name:
                return col

    return find_column(df, preferred)


def find_customer_column(df):
    preferred = [
        "customer_id",
        "customerid",
        "customer",
        "customers",
        "user_id",
        "user"
    ]

    for name in preferred:
        for col in df.columns:
            if str(col).lower().strip() == name:
                return col

    return find_column(
        df,
        [
            "customer_id",
            "customerid",
            "customer",
            "customers",
            "user_id",
            "user"
        ]
    )


def find_transaction_column(df):
    preferred = [
        "invoice",
        "invoice_no",
        "invoice_number",
        "invoice_id",
        "order_id",
        "order_number",
        "transaction_id",
        "transaction"
    ]

    for name in preferred:
        for col in df.columns:
            if str(col).lower().strip() == name:
                return col

    return find_column(df, preferred)


# ============================================================
# HELPERS
# ============================================================

def contains_any(text, keywords):
    text = str(text).lower()
    return any(keyword.lower() in text for keyword in keywords)


def contains_all(text, keywords):
    text = str(text).lower()
    return all(keyword.lower() in text for keyword in keywords)


def extract_number(text, default=5):
    match = re.search(r"\b(\d+)\b", str(text))

    if match:
        return int(match.group(1))

    return default


def clean_text(value):
    if pd.isna(value):
        return "Unknown"

    return str(value).strip()


def format_currency(value):
    try:
        return f"£{float(value):,.2f}"
    except Exception:
        return f"{value}"


def safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    )


def empty_result(message):
    return {
        "answer": message,
        "table": None,
        "chart_data": None,
        "chart_type": None,
        "x": None,
        "y": None,
        "title": None,
        "insight": None
    }


def make_result(
    answer,
    table=None,
    chart_type=None,
    x=None,
    y=None,
    title=None,
    insight=None
):
    """
    Standard result format.

    Both `table` and `chart_data` are returned so the dashboard
    can display a dataframe and generate a chart from the same
    analysis result.
    """

    return {
        "answer": answer,
        "table": table,
        "chart_data": table,
        "chart_type": chart_type,
        "x": x,
        "y": y,
        "title": title,
        "insight": insight
    }


def _clean_group_column(df, column):
    df = df.copy()

    if column is not None:
        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    return df


def _prepare_revenue(df, revenue_col):
    if revenue_col is None:
        return None

    return safe_numeric(
        df[revenue_col]
    ).fillna(0)


# ============================================================
# PRODUCT ANALYSIS
# ============================================================

def product_analysis(df, question):

    product_col = find_product_column(df)
    revenue_col = find_revenue_column(df)
    quantity_col = find_quantity_column(df)

    if product_col is None:
        return None

    work = _clean_group_column(
        df,
        product_col
    )

    # --------------------------------------------------------
    # TOP N PRODUCTS BY REVENUE
    # --------------------------------------------------------

    revenue_keywords = [
        "revenue",
        "sales",
        "money",
        "earning",
        "earnings"
    ]

    top_keywords = [
        "top",
        "highest",
        "best",
        "largest",
        "most revenue",
        "best selling"
    ]

    if (
        revenue_col is not None
        and contains_any(question, top_keywords)
        and contains_any(question, revenue_keywords)
    ):

        n = max(
            1,
            min(
                extract_number(question, 5),
                50
            )
        )

        work["_revenue"] = _prepare_revenue(
            work,
            revenue_col
        )

        result = (
            work
            .groupby(product_col)["_revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(n)
            .reset_index()
        )

        result.columns = [
            "Product",
            "Revenue"
        ]

        if result.empty:
            return empty_result(
                "No product revenue data was available."
            )

        top_product = result.iloc[0]

        if n == 1:
            answer = (
                f"The highest-revenue product is "
                f"**{top_product['Product']}**, generating "
                f"**{format_currency(top_product['Revenue'])}**."
            )
        else:
            answer = (
                f"Here are the **top {n} products by revenue**. "
                f"The leading product is "
                f"**{top_product['Product']}** with "
                f"**{format_currency(top_product['Revenue'])}**."
            )

        display_table = result.copy()

        display_table["Revenue"] = display_table[
            "Revenue"
        ].round(2)

        return make_result(
            answer=answer,
            table=display_table,
            chart_type="bar",
            x="Product",
            y="Revenue",
            title=f"Top {n} Products by Revenue",
            insight=(
                f"{top_product['Product']} is the largest "
                f"revenue contributor among the products "
                f"in this dataset."
            )
        )

    # --------------------------------------------------------
    # HIGHEST QUANTITY PRODUCT
    # --------------------------------------------------------

    quantity_keywords = [
        "quantity",
        "units",
        "sold",
        "volume"
    ]

    quantity_top_keywords = [
        "highest",
        "most",
        "top",
        "largest",
        "best selling"
    ]

    if (
        quantity_col is not None
        and contains_any(question, quantity_keywords)
        and contains_any(question, quantity_top_keywords)
    ):

        work["_quantity"] = safe_numeric(
            work[quantity_col]
        ).fillna(0)

        result = (
            work
            .groupby(product_col)["_quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        result.columns = [
            "Product",
            "Quantity"
        ]

        if result.empty:
            return empty_result(
                "No quantity data was available."
            )

        best = result.iloc[0]

        return make_result(
            answer=(
                f"**{best['Product']}** has the highest "
                f"quantity sold with "
                f"**{best['Quantity']:,.0f} units**."
            ),
            table=result,
            chart_type="bar",
            x="Product",
            y="Quantity",
            title="Top Products by Quantity Sold",
            insight=(
                f"{best['Product']} has the highest "
                f"unit volume in the dataset."
            )
        )

    # --------------------------------------------------------
    # HIGH QUANTITY + LOW REVENUE
    # --------------------------------------------------------

    if (
        revenue_col is not None
        and quantity_col is not None
        and contains_any(
            question,
            [
                "high quantity",
                "high quantities",
                "low revenue",
                "high quantity but low revenue",
                "quantity but low revenue",
                "high volume low revenue"
            ]
        )
    ):

        work["_revenue"] = _prepare_revenue(
            work,
            revenue_col
        )

        work["_quantity"] = safe_numeric(
            work[quantity_col]
        ).fillna(0)

        grouped = (
            work
            .groupby(product_col)
            .agg(
                Revenue=("_revenue", "sum"),
                Quantity=("_quantity", "sum")
            )
            .reset_index()
        )

        if grouped.empty:
            return empty_result(
                "No product data was available."
            )

        revenue_median = grouped["Revenue"].median()
        quantity_median = grouped["Quantity"].median()

        result = grouped[
            (grouped["Quantity"] >= quantity_median)
            & (grouped["Revenue"] <= revenue_median)
        ].copy()

        result = (
            result
            .sort_values(
                "Quantity",
                ascending=False
            )
            .head(10)
        )

        result.columns = [
            "Product",
            "Revenue",
            "Quantity"
        ]

        if result.empty:
            return empty_result(
                "No products matched the high-quantity "
                "and low-revenue criteria."
            )

        return make_result(
            answer=(
                "These products have relatively high unit "
                "volume but relatively low revenue compared "
                "with the dataset median."
            ),
            table=result,
            chart_type="bar",
            x="Product",
            y="Quantity",
            title="High Quantity but Lower Revenue Products",
            insight=(
                "These products may deserve investigation "
                "because strong unit volume is not translating "
                "into equally strong revenue."
            )
        )

    # --------------------------------------------------------
    # LOW REVENUE PRODUCTS
    # --------------------------------------------------------

    if (
        revenue_col is not None
        and contains_any(
            question,
            [
                "lowest revenue",
                "low revenue products",
                "worst revenue",
                "least revenue"
            ]
        )
    ):

        n = max(
            1,
            min(
                extract_number(question, 10),
                50
            )
        )

        work["_revenue"] = _prepare_revenue(
            work,
            revenue_col
        )

        result = (
            work
            .groupby(product_col)["_revenue"]
            .sum()
            .sort_values()
            .head(n)
            .reset_index()
        )

        result.columns = [
            "Product",
            "Revenue"
        ]

        if result.empty:
            return empty_result(
                "No low-revenue product data was available."
            )

        return make_result(
            answer=(
                f"These are the **bottom {n} products "
                f"by revenue**."
            ),
            table=result,
            chart_type="bar",
            x="Product",
            y="Revenue",
            title=f"Bottom {n} Products by Revenue",
            insight=(
                "These products generate the lowest revenue "
                "among the identifiable products in the dataset."
            )
        )

    return None


# ============================================================
# REVENUE ANALYSIS
# ============================================================

def revenue_analysis(df, question):

    revenue_col = find_revenue_column(df)

    if revenue_col is None:
        return None

    revenue = safe_numeric(
        df[revenue_col]
    ).fillna(0)

    total_revenue = revenue.sum()

    # --------------------------------------------------------
    # TOTAL REVENUE
    # --------------------------------------------------------

    if contains_any(
        question,
        [
            "total revenue",
            "total sales",
            "how much revenue",
            "how much sales",
            "revenue generated",
            "sales generated"
        ]
    ):

        return make_result(
            answer=(
                f"Total revenue is "
                f"**{format_currency(total_revenue)}**."
            ),
            insight=(
                f"The dataset contains "
                f"{format_currency(total_revenue)} "
                f"in total revenue."
            )
        )

    # --------------------------------------------------------
    # AVERAGE REVENUE
    # --------------------------------------------------------

    if contains_any(
        question,
        [
            "average revenue",
            "average sales",
            "mean revenue",
            "mean sales"
        ]
    ):

        average = revenue.mean()

        return make_result(
            answer=(
                f"Average revenue per row is "
                f"**{format_currency(average)}**."
            ),
            insight=None
        )

    # --------------------------------------------------------
    # TOP 10 REVENUE CONTRIBUTION
    # --------------------------------------------------------

    if (
        contains_any(
            question,
            [
                "percentage",
                "percent",
                "contribution",
                "share"
            ]
        )
        and contains_any(
            question,
            [
                "top 10",
                "top ten"
            ]
        )
    ):

        product_col = find_product_column(df)

        if product_col is not None:

            grouped = (
                df.assign(
                    _revenue=revenue
                )
                .groupby(product_col)["_revenue"]
                .sum()
                .sort_values(ascending=False)
            )

            top_10 = grouped.head(10).sum()

            if total_revenue != 0:

                percentage = (
                    top_10 / total_revenue
                ) * 100

                return make_result(
                    answer=(
                        f"The top 10 products contribute "
                        f"**{percentage:.2f}%** of total revenue."
                    ),
                    insight=(
                        "A high percentage indicates that revenue "
                        "is concentrated among a relatively small "
                        "number of products."
                    )
                )

    return None


# ============================================================
# MONTHLY / DATE ANALYSIS
# ============================================================

def monthly_analysis(df, question):

    date_col = find_month_column(df)
    revenue_col = find_revenue_column(df)

    if date_col is None or revenue_col is None:
        return None

    dates = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    revenue = safe_numeric(
        df[revenue_col]
    ).fillna(0)

    work = pd.DataFrame({
        "Date": dates,
        "Revenue": revenue
    }).dropna(
        subset=["Date"]
    )

    if work.empty:
        return None

    work["Month"] = (
        work["Date"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        work
        .groupby("Month")["Revenue"]
        .sum()
        .reset_index()
    )

    monthly.columns = [
        "Month",
        "Revenue"
    ]

    # --------------------------------------------------------
    # MONTHLY REVENUE
    # --------------------------------------------------------

    if contains_any(
        question,
        [
            "monthly revenue",
            "revenue trend",
            "monthly sales",
            "sales trend",
            "revenue by month",
            "revenue over time",
            "sales over time",
            "monthly performance"
        ]
    ):

        if len(monthly) >= 2:

            first = monthly["Revenue"].iloc[0]
            last = monthly["Revenue"].iloc[-1]

            if first != 0:

                change = (
                    (last - first)
                    / abs(first)
                ) * 100

                if change > 5:
                    trend = "increasing"
                elif change < -5:
                    trend = "decreasing"
                else:
                    trend = "relatively stable"

            else:
                trend = "not clearly measurable"

        else:
            trend = "not enough historical data"

        return make_result(
            answer=(
                f"The monthly revenue trend is "
                f"**{trend}**."
            ),
            table=monthly,
            chart_type="line",
            x="Month",
            y="Revenue",
            title="Monthly Revenue Trend",
            insight=(
                f"Revenue is **{trend}** across the "
                f"available monthly observations."
            )
        )

    # --------------------------------------------------------
    # HIGHEST REVENUE MONTH
    # --------------------------------------------------------

    if contains_any(
        question,
        [
            "highest month",
            "best month",
            "month highest",
            "highest revenue month",
            "which month had the highest",
            "which month generated the most",
            "best performing month"
        ]
    ):

        best = monthly.loc[
            monthly["Revenue"].idxmax()
        ]

        return make_result(
            answer=(
                f"The highest-revenue month is "
                f"**{best['Month']}**, with revenue of "
                f"**{format_currency(best['Revenue'])}**."
            ),
            table=monthly,
            chart_type="line",
            x="Month",
            y="Revenue",
            title="Monthly Revenue",
            insight=(
                f"{best['Month']} generated the highest "
                f"revenue among the available months."
            )
        )

    # --------------------------------------------------------
    # LOWEST REVENUE MONTH
    # --------------------------------------------------------

    if contains_any(
        question,
        [
            "lowest revenue month",
            "worst month",
            "lowest month",
            "month with lowest revenue"
        ]
    ):

        worst = monthly.loc[
            monthly["Revenue"].idxmin()
        ]

        return make_result(
            answer=(
                f"The lowest-revenue month is "
                f"**{worst['Month']}**, with revenue of "
                f"**{format_currency(worst['Revenue'])}**."
            ),
            table=monthly,
            chart_type="line",
            x="Month",
            y="Revenue",
            title="Monthly Revenue",
            insight=(
                f"{worst['Month']} generated the lowest "
                f"revenue among the available months."
            )
        )

    return None


# ============================================================
# COUNTRY ANALYSIS
# ============================================================

def country_analysis(df, question):

    country_col = find_country_column(df)
    revenue_col = find_revenue_column(df)

    if country_col is None:
        return None

    if revenue_col is not None and contains_any(
        question,
        [
            "country",
            "countries",
            "market",
            "region"
        ]
    ):

        revenue = safe_numeric(
            df[revenue_col]
        ).fillna(0)

        work = df.copy()

        work[country_col] = (
            work[country_col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

        result = (
            work.assign(
                _revenue=revenue
            )
            .groupby(country_col)["_revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        result.columns = [
            "Country",
            "Revenue"
        ]

        if result.empty:
            return None

        best = result.iloc[0]

        return make_result(
            answer=(
                f"**{best['Country']}** generated the "
                f"highest revenue at "
                f"**{format_currency(best['Revenue'])}**."
            ),
            table=result,
            chart_type="bar",
            x="Country",
            y="Revenue",
            title="Top Countries by Revenue",
            insight=(
                f"{best['Country']} is the largest "
                f"revenue market in this dataset."
            )
        )

    return None


# ============================================================
# QUANTITY ANALYSIS
# ============================================================

def quantity_analysis(df, question):

    quantity_col = find_quantity_column(df)

    if quantity_col is None:
        return None

    quantity = safe_numeric(
        df[quantity_col]
    ).fillna(0)

    total_quantity = quantity.sum()

    if contains_any(
        question,
        [
            "total quantity",
            "total units",
            "total units sold",
            "how many units",
            "units sold in total",
            "total volume"
        ]
    ):

        return make_result(
            answer=(
                f"Total quantity sold is "
                f"**{total_quantity:,.0f} units**."
            ),
            insight=(
                f"The dataset records approximately "
                f"{total_quantity:,.0f} units."
            )
        )

    return None


# ============================================================
# CUSTOMER ANALYSIS
# ============================================================

def customer_analysis(df, question):

    customer_col = find_customer_column(df)

    if customer_col is None:
        return None

    if contains_any(
        question,
        [
            "customer count",
            "number of customers",
            "how many customers",
            "unique customers",
            "customer base"
        ]
    ):

        unique_customers = (
            df[customer_col]
            .dropna()
            .nunique()
        )

        return make_result(
            answer=(
                f"There are **{unique_customers:,} unique "
                f"customers** in the dataset."
            ),
            insight=(
                f"The identifiable customer base contains "
                f"{unique_customers:,} unique customers."
            )
        )

    return None


# ============================================================
# AVERAGE ORDER VALUE
# ============================================================

def order_value_analysis(df, question):

    revenue_col = find_revenue_column(df)

    if revenue_col is None:
        return None

    if not contains_any(
        question,
        [
            "average order value",
            "aov",
            "average order",
            "order value"
        ]
    ):
        return None

    transaction_col = find_transaction_column(df)

    if transaction_col is not None:

        orders = (
            df[transaction_col]
            .dropna()
            .astype(str)
            .nunique()
        )

    else:

        orders = len(df)

    revenue = safe_numeric(
        df[revenue_col]
    ).fillna(0).sum()

    if orders == 0:
        return empty_result(
            "I could not calculate Average Order Value."
        )

    aov = revenue / orders

    return make_result(
        answer=(
            f"Average Order Value is "
            f"**{format_currency(aov)}**."
        ),
        insight=(
            f"The calculated average order value is "
            f"{format_currency(aov)}."
        )
    )


# ============================================================
# TRANSACTION / ORDER COUNT
# ============================================================

def transaction_analysis(df, question):

    transaction_col = find_transaction_column(df)

    if transaction_col is None:
        return None

    if not contains_any(
        question,
        [
            "number of orders",
            "how many orders",
            "total orders",
            "order count",
            "number of transactions",
            "how many transactions",
            "total transactions",
            "transaction count"
        ]
    ):
        return None

    orders = (
        df[transaction_col]
        .dropna()
        .astype(str)
        .nunique()
    )

    return make_result(
        answer=(
            f"The dataset contains **{orders:,} unique "
            f"orders/transactions**."
        ),
        insight=(
            f"{orders:,} unique transaction identifiers "
            f"were detected."
        )
    )


# ============================================================
# DATA QUALITY
# ============================================================

def data_quality_analysis(df, question):

    if contains_any(
        question,
        [
            "missing",
            "null",
            "blank",
            "data quality",
            "incomplete data"
        ]
    ):

        missing = df.isna().sum()

        total_missing = int(
            missing.sum()
        )

        missing_columns = (
            missing[missing > 0]
            .sort_values(ascending=False)
        )

        if total_missing == 0:

            answer = (
                "There are **no missing values** "
                "in the dataset."
            )

        else:

            details = ", ".join(
                [
                    f"{col}: {count:,}"
                    for col, count in missing_columns.items()
                ]
            )

            answer = (
                f"The dataset contains "
                f"**{total_missing:,} missing values**. "
                f"Main affected columns: {details}."
            )

        return make_result(
            answer=answer,
            insight=(
                "Missing values should be reviewed before "
                "using affected columns for modelling "
                "or decision-making."
                if total_missing > 0
                else "The dataset has no missing values."
            )
        )

    return None


# ============================================================
# DATASET SIZE
# ============================================================

def dataset_size_analysis(df, question):

    if contains_any(
        question,
        [
            "dataset size",
            "how many rows",
            "how many records",
            "number of rows",
            "number of columns",
            "dataset dimensions",
            "how large is the dataset",
            "dataset records"
        ]
    ):

        return make_result(
            answer=(
                f"The dataset contains **{len(df):,} rows** "
                f"and **{len(df.columns):,} columns**."
            ),
            insight=(
                f"Dataset dimensions are "
                f"{len(df):,} rows × {len(df.columns):,} columns."
            )
        )

    return None


# ============================================================
# AUTOMATIC BUSINESS INSIGHT
# ============================================================

def automatic_business_insight(df):

    revenue_col = find_revenue_column(df)
    product_col = find_product_column(df)

    if revenue_col is None:
        return (
            "The dataset does not contain an identifiable "
            "revenue column, so a revenue-based business "
            "insight cannot be generated."
        )

    revenue = safe_numeric(
        df[revenue_col]
    ).fillna(0)

    total_revenue = revenue.sum()

    if product_col is not None:

        work = df.copy()

        work[product_col] = (
            work[product_col]
            .fillna("Unknown")
            .astype(str)
        )

        grouped = (
            work.assign(
                _revenue=revenue
            )
            .groupby(product_col)["_revenue"]
            .sum()
            .sort_values(ascending=False)
        )

        if not grouped.empty:

            top_product = grouped.index[0]
            top_revenue = grouped.iloc[0]

            share = (
                top_revenue / total_revenue * 100
                if total_revenue != 0
                else 0
            )

            return (
                f"**{top_product}** is the leading revenue "
                f"product, contributing approximately "
                f"**{share:.2f}%** of total revenue."
            )

    return (
        f"The dataset contains "
        f"**{format_currency(total_revenue)}** "
        f"in total revenue."
    )


# ============================================================
# DATASET ANALYSIS
# ============================================================

def dataset_analysis(df):

    revenue_col = find_revenue_column(df)
    quantity_col = find_quantity_column(df)

    insights = []

    insights.append(
        f"The dataset contains **{len(df):,} rows** "
        f"and **{len(df.columns):,} columns**."
    )

    missing = int(
        df.isna().sum().sum()
    )

    if missing > 0:

        insights.append(
            f"There are **{missing:,} missing values** "
            f"that may require cleaning."
        )

    else:

        insights.append(
            "No missing values were detected."
        )

    if revenue_col is not None:

        revenue = safe_numeric(
            df[revenue_col]
        ).fillna(0)

        insights.append(
            f"Total revenue is "
            f"**{format_currency(revenue.sum())}**."
        )

    if quantity_col is not None:

        quantity = safe_numeric(
            df[quantity_col]
        ).fillna(0)

        insights.append(
            f"Total quantity is "
            f"**{quantity.sum():,.0f} units**."
        )

    transaction_col = find_transaction_column(df)

    if transaction_col is not None:

        transactions = (
            df[transaction_col]
            .dropna()
            .astype(str)
            .nunique()
        )

        insights.append(
            f"Total unique orders/transactions are "
            f"**{transactions:,}**."
        )

    customer_col = find_customer_column(df)

    if customer_col is not None:

        customers = (
            df[customer_col]
            .dropna()
            .nunique()
        )

        insights.append(
            f"Unique customers identified: "
            f"**{customers:,}**."
        )

    return make_result(
        answer="\n\n".join(
            [f"- {item}" for item in insights]
        ),
        insight=automatic_business_insight(df)
    )


# ============================================================
# QUESTION NORMALIZATION
# ============================================================

def normalize_question(question):

    question = str(
        question
    ).strip().lower()

    question = re.sub(
        r"\s+",
        " ",
        question
    )

    return question


# ============================================================
# MAIN AI ANALYST ENGINE
# ============================================================

def analyze_question(df, question):

    if df is None or df.empty:

        return empty_result(
            "The dataset is empty or unavailable."
        )

    if not question or not str(question).strip():

        return empty_result(
            "Please enter a question."
        )

    question = normalize_question(
        question
    )

    # --------------------------------------------------------
    # SPECIALISED ANALYZERS
    # --------------------------------------------------------

    analyzers = [
        lambda: product_analysis(df, question),
        lambda: revenue_analysis(df, question),
        lambda: monthly_analysis(df, question),
        lambda: country_analysis(df, question),
        lambda: quantity_analysis(df, question),
        lambda: customer_analysis(df, question),
        lambda: transaction_analysis(df, question),
        lambda: order_value_analysis(df, question),
        lambda: data_quality_analysis(df, question),
        lambda: dataset_size_analysis(df, question)
    ]

    for analyzer in analyzers:

        try:

            result = analyzer()

            if result is not None:
                return result

        except Exception:
            continue

    # --------------------------------------------------------
    # GENERIC DATASET ANALYSIS
    # --------------------------------------------------------

    if contains_any(
        question,
        [
            "analyze dataset",
            "analyse dataset",
            "give me insights",
            "business insights",
            "overview",
            "summarize",
            "summary",
            "analyze this data",
            "analyse this data",
            "what can you tell me"
        ]
    ):

        return dataset_analysis(df)

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return make_result(
        answer=(
            "I couldn't confidently understand that question "
            "from the available columns.\n\n"
            "Try questions such as:\n"
            "- Which product has the highest revenue?\n"
            "- Show top 5 products by revenue.\n"
            "- Which product sold the most units?\n"
            "- Which products have high quantity but low revenue?\n"
            "- What is the monthly revenue trend?\n"
            "- Which month had the highest revenue?\n"
            "- Which month had the lowest revenue?\n"
            "- Which country generated the most revenue?\n"
            "- What is the total revenue?\n"
            "- What is the average order value?\n"
            "- How many orders are there?\n"
            "- How many customers are there?\n"
            "- How many missing values are there?\n"
            "- Give me a summary of this dataset."
        ),
        insight=automatic_business_insight(df)
    )