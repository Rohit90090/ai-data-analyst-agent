from pathlib import Path
import pandas as pd

# Project root
BASE_DIR = Path(__file__).resolve().parents[1]

# Dataset path
file_path = BASE_DIR / "data" / "raw" / "online_retail_II.xlsx"

print("Loading dataset...")

# Read both sheets
df_2009 = pd.read_excel(file_path, sheet_name="Year 2009-2010")
df_2010 = pd.read_excel(file_path, sheet_name="Year 2010-2011")

# Combine both years
df = pd.concat([df_2009, df_2010], ignore_index=True)

print("\nDataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())