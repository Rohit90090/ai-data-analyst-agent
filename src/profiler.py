import pandas as pd
import numpy as np


def dataset_profile(df):
    """
    Generate a detailed dataset profile.
    """

    profile = []

    for column in df.columns:

        missing_count = int(df[column].isna().sum())

        missing_percentage = round(
            (missing_count / len(df)) * 100,
            2
        ) if len(df) > 0 else 0

        unique_count = int(df[column].nunique(dropna=True))

        data_type = str(df[column].dtype)

        profile.append({
            "Column": column,
            "Data Type": data_type,
            "Missing": missing_count,
            "Missing %": missing_percentage,
            "Unique Values": unique_count
        })

    return pd.DataFrame(profile)


def numeric_summary(df):
    """
    Generate statistical summary for numeric columns.
    """

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        return pd.DataFrame()

    summary = numeric_df.describe().T.reset_index()

    summary = summary.rename(
        columns={
            "index": "Column",
            "count": "Count",
            "mean": "Mean",
            "std": "Std Dev",
            "min": "Minimum",
            "25%": "25%",
            "50%": "Median",
            "75%": "75%",
            "max": "Maximum"
        }
    )

    return summary


def categorical_summary(df):
    """
    Generate summary for categorical columns.
    """

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    results = []

    for column in categorical_columns:

        results.append({
            "Column": column,
            "Unique Values": df[column].nunique(),
            "Most Common": (
                df[column].mode().iloc[0]
                if not df[column].mode().empty
                else "N/A"
            )
        })

    return pd.DataFrame(results)


def detect_outliers(df):
    """
    Detect potential outliers using IQR.
    """

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    results = []

    for column in numeric_columns:

        series = df[column].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = int(
            ((series < lower_bound) |
             (series > upper_bound)).sum()
        )

        results.append({
            "Column": column,
            "Outliers": outlier_count,
            "Outlier %": round(
                (outlier_count / len(series)) * 100,
                2
            )
        })

    return pd.DataFrame(results)