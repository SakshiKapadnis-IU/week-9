import pandas as pd
import numpy as np

class GroupEstimate:
    def __init__(self, estimate="mean"):
        # Strict validation
        if estimate not in ["mean", "median"]:
            raise ValueError("estimate must be either 'mean' or 'median'")
        self.estimate = estimate
        self.group_estimates = None
        self.group_cols = None

    def fit(self, X, y):
        """
        X: pandas DataFrame of categorical data
        y: 1D array-like of continuous values
        """
        # Convert to DataFrame and Series for safety
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        y = pd.Series(y)
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        # Combine X and y into a single DataFrame
        df = X.copy()
        df["__y__"] = y

        # Remember grouping columns
        self.group_cols = list(X.columns)

        # Compute group estimate
        if self.estimate == "mean":
            grouped = df.groupby(self.group_cols, dropna=False)["__y__"].mean()
        else:
            grouped = df.groupby(self.group_cols, dropna=False)["__y__"].median()

        # Save as DataFrame for merge lookup
        self.group_estimates = grouped.reset_index()

    def predict(self, X_):
        """
        X_: array-like or DataFrame of new categorical observations
        Returns: numpy array of predicted estimates
        """
        # Convert input to DataFrame if needed
        if not isinstance(X_, pd.DataFrame):
            X_ = pd.DataFrame(X_, columns=self.group_cols)

        # Merge with group estimates
        merged = pd.merge(X_, self.group_estimates, on=self.group_cols, how="left")

        # Count missing predictions
        missing_count = merged["__y__"].isna().sum()
        if missing_count > 0:
            print(f"{missing_count} observation(s) belong to missing group(s). Returning NaN for those.")

        # Always return numpy array (not list)
        return merged["__y__"].to_numpy()