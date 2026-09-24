from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DASHBOARD_DIR.parent

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "data" / "models"

BUSINESS_METRICS_FILE = DASHBOARD_DIR / "business_metrics.csv"
MONTHLY_METRICS_FILE = DASHBOARD_DIR / "monthly_business_metrics.csv"
PRODUCT_SUMMARY_FILE = DASHBOARD_DIR / "product_summary.csv"


# ============================================================
# BASIC CSV LOADER
# ============================================================

def _load_csv(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"\nFile not found:\n{file_path}\n\n"
            "Run this command first:\n"
            "python dashboard\\build_dashboard_cache.py"
        )

    return pd.read_csv(file_path)


# ============================================================
# BUSINESS METRICS
# ============================================================

def load_business_metrics():
    return _load_csv(BUSINESS_METRICS_FILE)


# ============================================================
# MONTHLY BUSINESS METRICS
# ============================================================

def load_monthly_metrics():

    df = _load_csv(MONTHLY_METRICS_FILE)

    if "Month" in df.columns:
        df["Month"] = pd.to_datetime(
            df["Month"],
            errors="coerce"
        )

    return df


# ============================================================
# PRODUCT SUMMARY
# ============================================================

def load_product_summary():

    df = _load_csv(PRODUCT_SUMMARY_FILE)

    if "StockCode" in df.columns:
        df["StockCode"] = (
            df["StockCode"]
            .astype(str)
            .str.strip()
        )

    return df


# ============================================================
# MONTHLY PRODUCT DEMAND
# ============================================================

def load_monthly_demand():

    file_path = (
        PROCESSED_DIR /
        "product_monthly_demand.csv"
    )

    df = _load_csv(file_path)

    if "Month" in df.columns:
        df["Month"] = pd.to_datetime(
            df["Month"],
            errors="coerce"
        )

    if "StockCode" in df.columns:
        df["StockCode"] = (
            df["StockCode"]
            .astype(str)
            .str.strip()
        )

    return df


# ============================================================
# DEMAND MODEL
# ============================================================

def load_demand_model():

    import joblib

    model_path = (
        MODELS_DIR /
        "best_demand_model.pkl"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"\nDemand model not found:\n{model_path}"
        )

    return joblib.load(model_path)


# ============================================================
# MODEL FEATURES
# ============================================================

def load_model_features():

    import joblib

    feature_path = (
        MODELS_DIR /
        "model_features.pkl"
    )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"\nModel features not found:\n{feature_path}"
        )

    return joblib.load(feature_path)


# ============================================================
# MODEL COMPARISON
# ============================================================

def load_model_comparison():

    file_path = (
        PROCESSED_DIR /
        "model_comparison.csv"
    )

    if not file_path.exists():
        return pd.DataFrame()

    return pd.read_csv(file_path)


# ============================================================
# DEMAND PREDICTIONS
# ============================================================

def load_demand_predictions():

    file_path = (
        PROCESSED_DIR /
        "demand_predictions.csv"
    )

    if not file_path.exists():
        return pd.DataFrame()

    return pd.read_csv(file_path)


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def get_product_summary(monthly_df=None):
    return load_product_summary()


def get_business_metrics(monthly_df=None):
    return load_business_metrics()


def get_monthly_metrics(monthly_df=None):
    return load_monthly_metrics()