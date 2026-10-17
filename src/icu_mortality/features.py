"""Feature engineering used before Lasso selection and in the final pipeline."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Vitals/labs where both extremes are dangerous ("Goldilocks" features).
# A squared term on a standardized value lets a linear model see a U-shape.
GOLDILOCKS_COLS = [
    "d1_heartrate_max", "d1_heartrate_min",
    "d1_sysbp_min", "d1_sysbp_max",
    "d1_mbp_min", "d1_mbp_max",
    "d1_temp_min", "d1_temp_max",
    "d1_resprate_min", "d1_resprate_max",
    "d1_spo2_min",
    "d1_potassium_max", "d1_potassium_min",
    "d1_glucose_max", "d1_glucose_min",
    "d1_sodium_max", "d1_sodium_min",
    "d1_wbc_max", "d1_wbc_min",
    "d1_hematocrit_max", "d1_hematocrit_min",
    "d1_calcium_max", "d1_calcium_min",
    "age", "weight",
]


class SquaredFeaturesOnly(BaseEstimator, TransformerMixin):
    """Append x**2 for every input column. No interaction terms.

    sklearn's PolynomialFeatures(degree=2) would add all pairwise products,
    which for 25 columns is 300+ extra features we didn't want Lasso to sift.
    """

    def fit(self, X, y=None):
        self.feature_names_in_ = getattr(
            X, "columns", [f"x{i}" for i in range(X.shape[1])]
        )
        return self

    def transform(self, X):
        X_array = X.values if hasattr(X, "values") else np.asarray(X)
        return np.hstack([X_array, X_array**2])

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        names = list(input_features)
        return np.array(names + [f"{n}^2" for n in names])


def drop_sparse_columns(df: pd.DataFrame, max_missing: float = 0.25) -> pd.DataFrame:
    """Drop columns whose missing fraction exceeds `max_missing` (186 -> 112 cols at 0.25)."""
    keep = df.columns[df.isnull().mean() <= max_missing]
    return df[keep]
