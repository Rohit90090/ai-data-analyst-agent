from pathlib import Path
import pandas as pd


def load_data(uploaded_file):
    """
    Load CSV or Excel file uploaded through Streamlit.
    """

    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            df = pd.read_excel(uploaded_file)

        else:
            raise ValueError(
                "Unsupported file format. Please upload CSV or Excel."
            )

        return df

    except Exception as e:
        raise RuntimeError(f"Could not load file: {e}")


def clean_column_names(df):
    """
    Standardize column names.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    return df


def get_basic_info(df):
    """
    Return basic dataset information.
    """

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_mb": round(
            df.memory_usage(deep=True).sum() / (1024 ** 2),
            2
        ),
    }