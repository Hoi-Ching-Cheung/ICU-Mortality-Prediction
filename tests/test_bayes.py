import numpy as np
import pytest

pm = pytest.importorskip("pymc")

from icu_mortality.bayes import build_model, posterior_mean_proba  # noqa: E402


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(42)
    n, n_hosp = 600, 3
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))]).astype("float32")
    hosp = rng.integers(0, n_hosp, n)
    true_beta = np.array([-1.0, 1.5, -0.8])
    y = rng.random(n) < 1 / (1 + np.exp(-(X @ true_beta)))
    return X, y.astype(int), hosp, true_beta


@pytest.mark.parametrize(
    "kind, beta_shape",
    [("pooled", (3,)), ("unpooled", (3, 3)), ("partial_pooled", (3, 3))],
)
def test_coefficient_shapes(toy_data, kind, beta_shape):
    X, y, hosp, _ = toy_data
    model = build_model(kind, X, y, hospital_idx=hosp)
    assert model.initial_point()["beta"].shape == beta_shape


def test_partial_pooling_has_hyperpriors(toy_data):
    X, y, hosp, _ = toy_data
    names = {v.name for v in build_model("partial_pooled", X, y, hosp).free_RVs}
    assert names == {"mu_beta", "sigma_beta", "beta"}


def test_hierarchical_models_require_hospital_index(toy_data):
    X, y, _, _ = toy_data
    with pytest.raises(ValueError, match="hospital_idx"):
        build_model("unpooled", X, y)


def test_pooled_map_recovers_true_coefficients(toy_data):
    X, y, _, true_beta = toy_data
    with build_model("pooled", X, y):
        est = pm.find_MAP(progressbar=False)["beta"]
    # The N(0, 1) prior shrinks a little; signs and rough size should survive.
    np.testing.assert_allclose(est, true_beta, atol=0.35)


def test_posterior_mean_proba_uses_each_rows_hospital():
    import xarray as xr
    from types import SimpleNamespace

    # Hospital 0 has a huge positive slope, hospital 1 a huge negative one.
    beta = np.array([[[[5.0], [-5.0]]]])  # chain, draw, hospital, feature
    trace = SimpleNamespace(
        posterior={"beta": xr.DataArray(beta, dims=["chain", "draw", "h", "f"])}
    )
    p = posterior_mean_proba(trace, np.ones((2, 1)), hospital_idx=np.array([0, 1]))
    assert p[0] > 0.99 and p[1] < 0.01
