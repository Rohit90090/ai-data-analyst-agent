from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DASHBOARD_DIR.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


RAW_FILE = RAW_DIR / "online_retail_II.xlsx"


BUSINESS_FILE = DASHBOARD_DIR / "business_metrics.csv"
MONTHLY_FILE = DASHBOARD_DIR / "monthly_business_metrics.csv"
PRODUCT_FILE = DASHBOARD_DIR / "product_summary.csv"


# ============================================================
# FIND CLEANED DATASET
# ============================================================

def find_cleaned_csv():

    if not PROCESSED_DIR.exists():
        return None

    candidates = list(
        PROCESSED_DIR.glob("*.csv")
    )

    required_columns = {
        "Invoice",
        "StockCode",
        "Quantity",
        "Price",
    }

    for file_path in candidates:

        try:

            sample = pd.read_csv(
                file_path,
                nrows=5
            )

            columns = set(
                sample.columns
            )

            if required_columns.issubset(columns):

                return file_path

        except Exception:
            continue

    return None


# ============================================================
# LOAD RAW EXCEL
# ============================================================

def load_raw_data():

    if not RAW_FILE.exists():

        raise FileNotFoundError(
            f"""
Raw dataset not found:

{RAW_FILE}

Make sure this file exists:

data/raw/online_retail_II.xlsx
"""
        )

    print("Loading UCI Online Retail II...")

    excel = pd.ExcelFile(
        RAW_FILE
    )

    print(
        "Available sheets:",
        excel.sheet_names
    )

    frames = []

    for sheet in excel.sheet_names:

        print(
            f"Reading sheet: {sheet}"
        )

        temp = pd.read_excel(
            RAW_FILE,
            sheet_name=sheet
        )

        frames.append(temp)

    df = pd.concat(
        frames,
        ignore_index=True
    )

    return df


# ============================================================
# CLEAN RAW DATA
# ============================================================

def clean_data(df):

    print(
        f"Raw rows: {len(df):,}"
    )

    df = df.copy()

    # --------------------------------------------------------
    # Standardize column names
    # --------------------------------------------------------

    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    # --------------------------------------------------------
    # Remove cancelled invoices
    # --------------------------------------------------------

    if "Invoice" in df.columns:

        invoice_text = (
            df["Invoice"]
            .astype(str)
            .str.strip()
        )

        cancelled = invoice_text.str.upper().str.startswith("C")

        df = df[
            ~cancelled
        ].copy()

    # --------------------------------------------------------
    # Customer ID
    # --------------------------------------------------------

    if "Customer ID" in df.columns:

        df = df[
            df["Customer ID"].notna()
        ].copy()

    # --------------------------------------------------------
    # Quantity
    # --------------------------------------------------------

    if "Quantity" in df.columns:

        df["Quantity"] = pd.to_numeric(
            df["Quantity"],
            errors="coerce"
        )

        df = df[
            df["Quantity"] > 0
        ].copy()

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    if "Price" in df.columns:

        df["Price"] = pd.to_numeric(
            df["Price"],
            errors="coerce"
        )

        df = df[
            df["Price"] > 0
        ].copy()

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    if "InvoiceDate" in df.columns:

        df["InvoiceDate"] = pd.to_datetime(
            df["InvoiceDate"],
            errors="coerce"
        )

        df = df[
            df["InvoiceDate"].notna()
        ].copy()

    # --------------------------------------------------------
    # Revenue
    # --------------------------------------------------------

    df["Revenue"] = (
        df["Quantity"] *
        df["Price"]
    )

    # --------------------------------------------------------
    # Clean Customer ID
    # --------------------------------------------------------

    if "Customer ID" in df.columns:

        df["Customer ID"] = pd.to_numeric(
            df["Customer ID"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # StockCode
    # --------------------------------------------------------

    if "StockCode" in df.columns:

        df["StockCode"] = (
            df["StockCode"]
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # Description
    # --------------------------------------------------------

    if "Description" in df.columns:

        df["Description"] = (
            df["Description"]
            .fillna("Unknown Product")
            .astype(str)
            .str.strip()
        )

    print(
        f"Clean rows: {len(df):,}"
    )

    return df


# ============================================================
# BUILD BUSINESS METRICS
# ============================================================

def build_business_metrics(df):

    metrics = {

        "Revenue": df["Revenue"].sum(),

        "Quantity": df["Quantity"].sum(),

        "Customers": (
            df["Customer ID"]
            .nunique()
        ),

        "Transactions": (
            df["Invoice"]
            .nunique()
        ),

        "Products": (
            df["StockCode"]
            .nunique()
        ),
    }

    result = pd.DataFrame(
        {
            "Metric": list(
                metrics.keys()
            ),
            "Value": list(
                metrics.values()
            ),
        }
    )

    result.to_csv(
        BUSINESS_FILE,
        index=False
    )

    print(
        f"Created: {BUSINESS_FILE}"
    )

    return result


# ============================================================
# BUILD MONTHLY METRICS
# ============================================================

def build_monthly_metrics(df):

    temp = df.copy()

    temp["Month"] = (
        temp["InvoiceDate"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly = (
        temp
        .groupby("Month")
        .agg(
            Revenue=("Revenue", "sum"),
            Quantity=("Quantity", "sum"),
            Transactions=("Invoice", "nunique"),
            Customers=("Customer ID", "nunique"),
        )
        .reset_index()
        .sort_values("Month")
    )

    monthly.to_csv(
        MONTHLY_FILE,
        index=False
    )

    print(
        f"Created: {MONTHLY_FILE}"
    )

    return monthly


# ============================================================
# BUILD PRODUCT SUMMARY
# ============================================================

def build_product_summary(df):

    aggregation = {

        "Revenue": (
            "Revenue",
            "sum"
        ),

        "Quantity": (
            "Quantity",
            "sum"
        ),

        "Transactions": (
            "Invoice",
            "nunique"
        ),

        "Customers": (
            "Customer ID",
            "nunique"
        ),

    }

    if "Description" in df.columns:

        aggregation[
            "Description"
        ] = (
            "Description",
            "first"
        )

    product = (
        df
        .groupby("StockCode")
        .agg(**aggregation)
        .reset_index()
    )

    product = product.sort_values(
        "Revenue",
        ascending=False
    )

    product.to_csv(
        PRODUCT_FILE,
        index=False
    )

    print(
        f"Created: {PRODUCT_FILE}"
    )

    return product


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BUILDING DASHBOARD CACHE")
    print("=" * 60)

    cleaned_file = find_cleaned_csv()

    # --------------------------------------------------------
    # Use existing cleaned CSV if available
    # --------------------------------------------------------

    if cleaned_file is not None:

        print(
            f"\nUsing cleaned dataset:\n{cleaned_file}"
        )

        df = pd.read_csv(
            cleaned_file
        )

        # Date conversion
        if "InvoiceDate" in df.columns:

            df["InvoiceDate"] = pd.to_datetime(
                df["InvoiceDate"],
                errors="coerce"
            )

        # Make sure Revenue exists
        if "Revenue" not in df.columns:

            df["Revenue"] = (
                pd.to_numeric(
                    df["Quantity"],
                    errors="coerce"
                )
                *
                pd.to_numeric(
                    df["Price"],
                    errors="coerce"
                )
            )

    # --------------------------------------------------------
    # Otherwise load raw UCI dataset
    # --------------------------------------------------------

    else:

        print(
            "\nNo suitable cleaned CSV found."
        )

        print(
            "Loading raw UCI Online Retail II..."
        )

        df = load_raw_data()

        df = clean_data(df)

    # --------------------------------------------------------
    # Required columns check
    # --------------------------------------------------------

    required = [
        "Invoice",
        "StockCode",
        "Quantity",
        "Price",
        "Customer ID",
        "InvoiceDate",
        "Revenue",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"""
Required columns are missing:

{missing}

Available columns:

{list(df.columns)}
"""
        )

    # --------------------------------------------------------
    # Build files
    # --------------------------------------------------------

    print("\nBuilding business metrics...")

    business = build_business_metrics(
        df
    )

    print("\nBuilding monthly metrics...")

    monthly = build_monthly_metrics(
        df
    )

    print("\nBuilding product summary...")

    products = build_product_summary(
        df
    )

    # --------------------------------------------------------
    # Display verification
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CACHE BUILD COMPLETE")
    print("=" * 60)

    print("\nBUSINESS METRICS")

    print(
        business.to_string(
            index=False
        )
    )

    print("\nFILES CREATED:")

    print(
        BUSINESS_FILE
    )

    print(
        MONTHLY_FILE
    )

    print(
        PRODUCT_FILE
    )

    print("\nExpected business totals approximately:")

    print(
        f"Revenue:      £{df['Revenue'].sum():,.2f}"
    )

    print(
        f"Quantity:     {df['Quantity'].sum():,.0f}"
    )

    print(
        f"Customers:    {df['Customer ID'].nunique():,}"
    )

    print(
        f"Transactions: {df['Invoice'].nunique():,}"
    )

    print(
        f"Products:     {df['StockCode'].nunique():,}"
    )


if __name__ == "__main__":
    main()