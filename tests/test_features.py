import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from icu_mortality.features import SquaredFeaturesOnly, drop_sparse_columns


def test_squares_without_interactions():
    X = pd.DataFrame({"a": [1.0, -2.0, 3.0], "b": [0.5, 0.0, -1.0]})
    out = SquaredFeaturesOnly().fit_transform(X)

    assert out.shape == (3, 4)  # 2 originals + 2 squares, no a*b
    np.testing.assert_allclose(out[:, :2], X.values)
    np.testing.assert_allclose(out[:, 2:], X.values**2)


def test_feature_names_round_trip_through_column_transformer():
    # Lasso coefficients are mapped back to clinical names via these strings,
    # so a broken name here silently breaks feature selection downstream.
    X = pd.DataFrame({"d1_temp_min": [36.0, 38.5, 35.0], "age": [40, 70, 85]})
    ct = ColumnTransformer(
        [("poly", Pipeline([("s", StandardScaler()), ("sq", SquaredFeaturesOnly())]),
          ["d1_temp_min", "age"])]
    )
    ct.fit(X)
    assert list(ct.get_feature_names_out()) == [
        "poly__d1_temp_min", "poly__age", "poly__d1_temp_min^2", "poly__age^2",
    ]


def test_squared_term_captures_u_shape():
    # After standardizing, both hypo- and hyperthermia map to a large square.
    temps = pd.DataFrame({"t": [34.0, 37.0, 40.0]})
    z = StandardScaler().fit_transform(temps)
    sq = SquaredFeaturesOnly().fit_transform(z)[:, 1]
    assert sq[0] == sq[2] > sq[1]


def test_drop_sparse_columns_threshold_is_inclusive():
    df = pd.DataFrame({
        "full": [1, 2, 3, 4],
        "quarter": [1, None, 3, 4],      # 25% missing -> kept
        "half": [1, None, None, 4],      # 50% missing -> dropped
    })
    assert list(drop_sparse_columns(df, 0.25).columns) == ["full", "quarter"]
