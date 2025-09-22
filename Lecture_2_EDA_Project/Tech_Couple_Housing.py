import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

df = pd.read_csv('KC_housing_data.csv')

# Display basic information
print("--- Dataset Info ---")
df.info()

# Display the first few rows
print("\n--- Head of Dataset ---")
print(df.head())

# Display descriptive statistics for numerical columns
print("\n--- Descriptive Statistics (Numerical) ---")
print(df.describe())

# Display value counts for categorical columns (adjust as needed for your data)
print("\n--- Value Counts (Categorical) ---")
for column in df.select_dtypes(include='object').columns:
    print(f"\n{column}:\n{df[column].value_counts()}")

# Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Identify numerical and categorical features
numerical_features = df.select_dtypes(include=np.number).columns.tolist()
categorical_features = df.select_dtypes(include='object').columns.tolist()
print(f"\nNumerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")