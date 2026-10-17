"""Bayesian logistic regression: pooled, unpooled and partially pooled by hospital.

The three models differ only in how coefficients are shared across hospitals:

    pooled           one beta vector for everyone
    unpooled         an independent beta vector per hospital
    partial_pooled   per-hospital betas drawn from a shared Normal(mu, sigma)
"""

import numpy as np
import pymc as pm


def build_model(kind: str, X, y, hospital_idx=None, features=None):
    """Return an (unsampled) PyMC model.

    X should be standardized and include an intercept column.
    hospital_idx maps each row to 0..n_hospitals-1 (needed unless kind == "pooled").
    """
    n_obs, n_feat = X.shape
    coords = {
        "obs_id": np.arange(n_obs),
        "features": features if features is not None else [f"x{i}" for i in range(n_feat)],
    }
    if kind != "pooled":
        if hospital_idx is None:
            raise ValueError(f"{kind} model needs hospital_idx")
        coords["hospital"] = np.arange(int(np.max(hospital_idx)) + 1)

    with pm.Model(coords=coords) as model:
        X_data = pm.Data("X", X, dims=("obs_id", "features"))

        if kind == "pooled":
            beta = pm.Normal("beta", mu=0, sigma=1, dims="features")
            logit_p = pm.math.dot(X_data, beta)
        elif kind == "unpooled":
            beta = pm.Normal("beta", mu=0, sigma=1, dims=("hospital", "features"))
            logit_p = pm.math.sum(X_data * beta[hospital_idx, :], axis=1)
        elif kind == "partial_pooled":
            mu_beta = pm.Normal("mu_beta", mu=0, sigma=1, dims="features")
            sigma_beta = pm.HalfNormal("sigma_beta", sigma=1, dims="features")
            beta = pm.Normal(
                "beta", mu=mu_beta, sigma=sigma_beta, dims=("hospital", "features")
            )
            logit_p = pm.math.sum(X_data * beta[hospital_idx, :], axis=1)
        else:
            raise ValueError(f"unknown model kind: {kind!r}")

        pm.Bernoulli("y", logit_p=logit_p, observed=y, dims="obs_id")

    return model


def posterior_mean_proba(trace, X, hospital_idx=None):
    """P(death) using the posterior-mean coefficients.

    A plug-in point estimate (same as the notebook), not the full
    posterior predictive.
    """
    beta = trace.posterior["beta"].mean(dim=["chain", "draw"]).values
    if beta.ndim == 1:
        logit = X @ beta
    else:
        logit = np.sum(X * beta[hospital_idx, :], axis=1)
    return 1.0 / (1.0 + np.exp(-logit))
