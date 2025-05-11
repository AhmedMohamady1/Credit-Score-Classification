
from sklearn.base import BaseEstimator, TransformerMixin
import re
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
"""
BaseEstimator, TransformerMixin: Inherits from scikit-learn base classes to make this compatible with scikit-learn pipelines.

BaseEstimator provides get_params() and set_params() methods, while TransformerMixin adds the fit_transform() method for easy chaining.
"""
# Convert Credit History Age String to Months
class CreditHistoryAgeToMonths(BaseEstimator, TransformerMixin):
    def __init__(self, column='Credit_History_Age', new_column_name='Credit_History_Age_in_months'):
        self.column = column
        self.new_column_name = new_column_name

    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        X = X.copy()
        if self.column not in X.columns:
            return X

        def to_months(x):
            if pd.isna(x):
                return np.nan
            try:
                s = str(x).strip()
                match = re.search(r'(\d+)\s*Years?\s*(?:and)?\s*(\d+)?\s*Months?', s, re.IGNORECASE)
                if match:
                    years = int(match.group(1)) if match.group(1) else 0
                    months = int(match.group(2)) if match.group(2) else 0
                    return years * 12 + months
                match_years_only = re.search(r'(\d+)\s*Years?', s, re.IGNORECASE)
                if match_years_only:
                    years = int(match_years_only.group(1))
                    return years * 12
                match_months_only = re.search(r'(\d+)\s*Months?', s, re.IGNORECASE)
                if match_months_only:
                    months = int(match_months_only.group(1))
                    return months

            except Exception: # Catch potential errors during conversion
                pass # Return NaN if any error occurs
            return np.nan # Return NaN if no match or error

        # Use .loc for safer assignment
        X.loc[:, self.new_column_name] = X[self.column].apply(to_months)
        return X

# Clean numeric columns: Remove special characters and convert to numeric
class CleanNumeric(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create a copy to avoid modifying the original data
        X = X.copy()

        # Apply cleaning to each specified column
        for c in self.columns:
            if c in X.columns:
                # Convert to string, remove special characters, and convert to numeric
                X[c] = pd.to_numeric(
                    X[c].astype(str).str.replace(r'[^0-9.]', '', regex=True),
                    errors='coerce' # handles empty and non-numeric data
                )
        return X

# Clean Category Strings
class CategoryCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns:
            if c in X.columns:
                # Ensure column is string type first
                X[c] = X[c].astype(str)
                # Apply cleaning steps
                X[c] = (
                    X[c]
                        .str.replace(r'[^A-Za-z\s]', '', regex=True)
                        .str.strip()
                        .str.replace(r'\s+', '_', regex=True)
                        .str.lower()
                        .replace(r'^_+$', np.nan, regex=True) # Handle cases that become only underscores ^start and end of string$
                        .replace(r'^\s*$', np.nan, regex=True) # Replace empty/whitespace-only with NaN
                        .replace('nan', np.nan) # Replace string 'nan' with NaN
                )
        return X

# Forward/backward fill by Customer_ID for static fields
class StaticFieldFiller(BaseEstimator, TransformerMixin):
    def __init__(self, columns, group_col='Customer_ID'):
        self.columns = columns
        self.group_col = group_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Check if group column exists
        if self.group_col not in X.columns:
            return X

        # Fill only the specified columns
        filled = X.groupby(self.group_col, group_keys=False)[self.columns].apply(lambda g: g.ffill().bfill())
        X[self.columns] = filled  # Update only the static columns

        return X

class LocalModeCatImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, window=3):
        self.columns = columns
        self.window = window

    def fit(self, X, y=None):
        # No fitting required for this imputer as it works row-wise without prior statistics
        return self

    def local_mode_impute(self, series):
        # Replace 'nan' strings with actual NaNs
        series = series.replace('nan', np.nan)
        # Create a copy to avoid modifying original data
        data = series.copy()
        n = len(data)
        for i in range(n):
            # Check if the value is NaN
            if pd.isna(data.iloc[i]):
                # Define the local window
                start = max(0, i - self.window // 2)
                end = min(n, i + self.window // 2 + 1)
                window_values = data[start:i].tolist() + data[i+1:end].tolist()
                # Use mode as the local fill value, excluding NaNs
                if window_values:
                    mode_value = pd.Series([v for v in window_values if not pd.isna(v)]).mode()
                    if len(mode_value) > 0:
                        data.iloc[i] = mode_value[0]
        return data

    def transform(self, X):
        # Apply the local mode imputation for each specified column
        X = X.copy()
        for col in self.columns:
            X[col] = self.local_mode_impute(X[col])
        return X

class MixedCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_features=None, nominal_features=None):
        self.ordinal_features = ordinal_features or []
        self.nominal_features = nominal_features or []
        self.label_encoders_ = {}

    def fit(self, X, y=None):
        # Fit LabelEncoders on ordinal features
        for col in self.ordinal_features:
            le = LabelEncoder()
            le.fit(X[col].astype(str).unique())  # Fit only on unique non-null values
            self.label_encoders_[col] = le
        return self

    def transform(self, X):
        X = X.copy()

        # Apply Label Encoding to ordinal features
        for col in self.ordinal_features:
            if col in X.columns:
                le = self.label_encoders_[col]
                # Encode known values, use -1 for unseen values
                """ Lambda Function Logic:
                le.transform([val]):
                LabelEncoder requires a list as input, even for a single value
                This returns a NumPy array with a single encoded value
                [0]:
                Extracts the single integer from the returned array """
                X[col] = X[col].astype(str).apply(lambda val: le.transform([val])[0] if val in le.classes_ else -1)

        # Apply One-Hot Encoding to nominal features
        if self.nominal_features:
            X = pd.get_dummies(X, columns=self.nominal_features, dummy_na=True)

        return X

# Drop Specified Columns
# for features afrer transformation and not used featuer in model
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure columns to drop actually exist
        cols_to_drop = [col for col in self.columns if col in X.columns]
        return X.drop(columns=cols_to_drop, errors='ignore')


# KNN and Z_score were not effecient bec of corrolation and bec of the data range for some feature we cannot use ordinal methods 
class MedianNumImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, outlier_factor=1.5):
        self.columns = columns
        self.outlier_factor = outlier_factor
        self.medians_ = {}
        self.iqr_bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for col in self.columns:
            Q1 = X_df[col].quantile(0.25)
            Q3 = X_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.outlier_factor * IQR
            upper_bound = Q3 + self.outlier_factor * IQR
            self.iqr_bounds_[col] = (lower_bound, upper_bound)
            self.medians_[col] = X_df[col].median()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in self.columns:
            if col in X_df.columns:
                median = self.medians_[col]
                lower, upper = self.iqr_bounds_[col]
                # Replace NaNs
                X_df[col] = X_df[col].fillna(median)
                # Replace outliers
                X_df[col] = X_df[col].mask((X_df[col] < lower) | (X_df[col] > upper), median)
        return X_df

class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.selected_features_ = []
        self.feature_names_in_ = None

    def fit(self, X, y):
        # Ensure input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = X.columns

        # Calculate correlation with target for each column
        correlations = X.apply(lambda col: np.corrcoef(col, y)[0, 1]) # [0, 1] for first row in cor matrix 
        abs_correlations = correlations.abs()

        # Select features with correlation above threshold
        self.selected_features_ = abs_correlations[abs_correlations >= self.threshold].index.tolist()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_features_)

class RoundDecimals(BaseEstimator, TransformerMixin):
    def __init__(self, decimals=2):
        self.decimals = decimals

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_rounded = X.copy()
        if isinstance(X_rounded, pd.DataFrame):
            X_rounded = X_rounded.round(self.decimals)
        else:
            X_rounded = np.round(X_rounded, self.decimals)
        return X_rounded

class DebugColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X