# threshold-ridge

Scikit-learn compatible Thresholded Ridge regression for high-dimensional variable selection.

---

## Overview

`threshold-ridge` implements the **Thresholded Ridge (TR)** estimator — a two-step procedure for sparse regression in high-dimensional settings ($p > n$):

1. Fit Ridge regression with regularisation parameter $h_n$.
2. Apply hard thresholding: retain only coefficients with $|\hat{\theta}_j| > a_n$.

The hyperparameters $(h_n, a_n)$ are jointly selected by one of two data-driven criteria:

- **BIC** (`tuning='bic'`): Bayesian Information Criterion.
- **LOOCV** (`tuning='loocv'`): Leave-One-Out CV proxy exploiting the Ridge hat-matrix diagonal.

The estimator is fully compatible with the scikit-learn API and can be used inside `Pipeline`, `GridSearchCV`, and `cross_val_score`.

---

## Installation

```bash
pip install threshold-ridge
```

Install the development version from GitHub:

```bash
pip install git+https://github.com/Farmer122/Threshold_Ridge.git
```

---

## Quick Start

```python
import numpy as np
from threshold_ridge import ThresholdedRidge

# Simulate a sparse high-dimensional dataset
rng = np.random.default_rng(0)
X = rng.standard_normal((200, 500))
beta = np.zeros(500)
beta[:25] = 1.0
y = X @ beta + rng.standard_normal(200)

# Fit with BIC tuning (default)
tr = ThresholdedRidge(tuning='bic')
tr.fit(X, y)

print(tr.support_.sum())   # number of selected variables
print(tr.coef_)            # sparse coefficient vector
print(tr.score(X, y))      # R-squared

# Use LOOCV tuning instead
tr_cv = ThresholdedRidge(tuning='loocv')
tr_cv.fit(X, y)

# Works inside a scikit-learn Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('tr', ThresholdedRidge(tuning='bic')),
])
pipe.fit(X, y)
```

---

## API Reference

### `ThresholdedRidge`

```python
ThresholdedRidge(
    tuning='bic',       # 'bic' or 'loocv'
    h_n_grid=None,      # Ridge penalty grid (default: logspace(-2, 2, 20))
    a_n_grid=None,      # Threshold grid (default: linspace(0.01, 0.6, 15))
    fit_intercept=True,
)
```

**Fitted attributes**

| Attribute | Description |
|---|---|
| `coef_` | Sparse coefficient vector of shape `(n_features,)` |
| `intercept_` | Fitted intercept (0 if `fit_intercept=False`) |
| `support_` | Boolean mask of selected features |
| `best_h_n_` | Optimal Ridge regularisation parameter |
| `best_a_n_` | Optimal hard-threshold value |

---

## Development

```bash
git clone https://github.com/Farmer122/Threshold_Ridge.git
cd threshold-ridge
pip install -e ".[dev]"
pytest tests/ -v
```

To build and publish a release to PyPI:

```bash
python -m build
twine upload dist/*
```

---

## References

- Hoerl, A. E. and Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1), 55--67.
- Schwarz, G. (1978). Estimating the Dimension of a Model. *Annals of Statistics*, 6(2), 461--464.
- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267--288.

---

## License

MIT License. See [LICENSE](LICENSE).
