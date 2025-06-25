import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """Loads data from CSV, JSON, or SAS files."""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    elif file_path.endswith('.sas7bdat'):
        try:
            df = pd.read_sas(file_path)
        except Exception as e:
            raise ConnectionError(f"Failed to read SAS file. Ensure 'saspy' is installed and configured if needed. Error: {e}")
    else:
        raise ValueError("Unsupported file type")
    
    # Clean up column names: remove spaces and special characters
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
    
    return df

def clean_and_preprocess_data(df):
    """Performs automated data cleaning and preprocessing in a generic way."""
    
    # --- NEW GENERALIZED CLEANING LOGIC ---
    # Attempt to convert object columns that look like numbers into numeric types.
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            # If conversion fails, it's genuinely a categorical column.
            pass
    # --- END OF NEW LOGIC ---

    # Separate numeric and categorical columns AFTER potential conversion
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Impute missing values
    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
    
    # Outlier detection using IQR
    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_condition = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))
        if outlier_condition.any():
            outliers[col] = df[outlier_condition].index.tolist()

    return df, {"outliers": outliers, "numeric_cols": numeric_cols, "categorical_cols": categorical_cols}