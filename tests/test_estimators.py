"""
Tests for threshold_ridge.ThresholdedRidge.

Run with:  pytest tests/ -v
"""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from threshold_ridge import ThresholdedRidge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sparse_data():
    """A simple sparse linear model: n=150, p=50, s=5."""
    rng = np.random.default_rng(42)
    n, p, s = 150, 50, 5
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:s] = 1.0
    y = X @ beta + rng.standard_normal(n)
    return X, y, beta


# ---------------------------------------------------------------------------
# Scikit-learn compatibility checks
# ---------------------------------------------------------------------------


@parametrize_with_checks(
    [ThresholdedRidge(tuning="bic"), ThresholdedRidge(tuning="loocv")]
)
def test_sklearn_compatible(estimator, check):
    """Verify the estimator passes the full sklearn estimator check suite."""
    check(estimator)


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tuning", ["bic", "loocv"])
def test_fit_returns_self(sparse_data, tuning):
    X, y, _ = sparse_data
    tr = ThresholdedRidge(tuning=tuning)
    assert tr.fit(X, y) is tr


@pytest.mark.parametrize("tuning", ["bic", "loocv"])
def test_coef_shape(sparse_data, tuning):
    X, y, _ = sparse_data
    tr = ThresholdedRidge(tuning=tuning).fit(X, y)
    assert tr.coef_.shape == (X.shape[1],)


@pytest.mark.parametrize("tuning", ["bic", "loocv"])
def test_predict_shape(sparse_data, tuning):
    X, y, _ = sparse_data
    tr = ThresholdedRidge(tuning=tuning).fit(X, y)
    assert tr.predict(X).shape == (X.shape[0],)


@pytest.mark.parametrize("tuning", ["bic", "loocv"])
def test_n_features_in(sparse_data, tuning):
    X, y, _ = sparse_data
    tr = ThresholdedRidge(tuning=tuning).fit(X, y)
    assert tr.n_features_in_ == X.shape[1]


# ---------------------------------------------------------------------------
# Sparsity and support
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tuning", ["bic", "loocv"])
def test_support_is_boolean_mask(sparse_data, tuning):
    X, y, _ = sparse_data
    tr = ThresholdedRidge(tuning=tuning).fit(X, y)
    assert tr.support_.dtype == bool
    assert tr.support_.shape == (X.shape[1],)


def test_support_is_strict_subset(sparse_data):
    """Selected support should be strictly smaller than p."""
    X, y, _ = sparse_data
    tr = ThresholdedRidge(tuning="bic").fit(X, y)
    assert 0 < int(tr.support_.sum()) < X.shape[1]


def test_coef_support_consistency(sparse_data):
    """Non-zero entries in coef_ must correspond exactly to support_."""
    X, y, _ = sparse_data
    tr = ThresholdedRidge(tuning="bic").fit(X, y)
    np.testing.assert_array_equal(tr.coef_ != 0, tr.support_)


# ---------------------------------------------------------------------------
# Intercept handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_fit_intercept_flag(sparse_data, fit_intercept):
    X, y, _ = sparse_data
    tr = ThresholdedRidge(tuning="bic", fit_intercept=fit_intercept).fit(X, y)
    if not fit_intercept:
        assert tr.intercept_ == 0.0
    else:
        assert isinstance(tr.intercept_, float)


# ---------------------------------------------------------------------------
# Tuned hyperparameter attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tuning", ["bic", "loocv"])
def test_best_params_stored(sparse_data, tuning):
    X, y, _ = sparse_data
    tr = ThresholdedRidge(tuning=tuning).fit(X, y)
    assert hasattr(tr, "best_h_n_")
    assert hasattr(tr, "best_a_n_")
    assert tr.best_h_n_ > 0
    assert tr.best_a_n_ >= 0


# ---------------------------------------------------------------------------
# Custom hyperparameter grids
# ---------------------------------------------------------------------------


def test_custom_grids(sparse_data):
    X, y, _ = sparse_data
    h_grid = np.logspace(-1, 1, 5)
    a_grid = np.linspace(0.1, 0.5, 5)
    tr = ThresholdedRidge(tuning="bic", h_n_grid=h_grid, a_n_grid=a_grid).fit(X, y)
    assert tr.coef_.shape == (X.shape[1],)
    assert tr.best_h_n_ in h_grid
    assert tr.best_a_n_ in a_grid


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_tuning(sparse_data):
    X, y, _ = sparse_data
    with pytest.raises(ValueError, match="tuning must be one of"):
        ThresholdedRidge(tuning="invalid").fit(X, y)


def test_predict_before_fit_raises():
    with pytest.raises(Exception):
        ThresholdedRidge().predict(np.ones((5, 3)))


# ---------------------------------------------------------------------------
# Pipeline compatibility
# ---------------------------------------------------------------------------


def test_pipeline_compatibility(sparse_data):
    X, y, _ = sparse_data
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("tr", ThresholdedRidge(tuning="bic")),
        ]
    )
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    assert y_pred.shape == (X.shape[0],)
