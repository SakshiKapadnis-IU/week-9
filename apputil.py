import pandas as pd
import numpy as np

class GroupEstimate:
    def __init__(self, estimate="mean"):
        # Validate estimate type
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
        # Check matching lengths
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        # Combine into single DataFrame
        df = X.copy()
        df["y"] = y

        # Remember which columns we grouped by
        self.group_cols = list(X.columns)

        # Calculate group-wise mean or median
        if self.estimate == "mean":
            self.group_estimates = df.groupby(self.group_cols)["y"].mean().reset_index()
        else:
            self.group_estimates = df.groupby(self.group_cols)["y"].median().reset_index()

    def predict(self, X_):
        """
        X_: array-like or DataFrame of new categorical observations
        """
        # Convert input to DataFrame if not already
        if not isinstance(X_, pd.DataFrame):
            X_ = pd.DataFrame(X_, columns=self.group_cols)

        # Merge new data with stored group estimates
        merged = pd.merge(X_, self.group_estimates, on=self.group_cols, how="left")

        # Check missing predictions
        missing_count = merged["y"].isna().sum()
        if missing_count > 0:
            print(f"{missing_count} observation(s) belong to missing group(s). Returning NaN for those.")

        return merged["y"].values