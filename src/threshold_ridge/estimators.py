"""
Scikit-learn compatible estimator class for Thresholded Ridge regression.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._tuning import tune_bic, tune_loocv

_TUNING_STRATEGIES = {
    "bic": tune_bic,
    "loocv": tune_loocv,
}


class ThresholdedRidge(BaseEstimator, RegressorMixin):
    """
    Thresholded Ridge regression for high-dimensional variable selection.

    Fits a Ridge regression and applies a hard-thresholding operator to the
    resulting coefficient vector, yielding a sparse estimate.  The Ridge
    regularisation parameter ``h_n`` and the threshold ``a_n`` are jointly
    selected over supplied grids using one of two data-driven criteria:

    - ``'bic'``:   Bayesian Information Criterion.
    - ``'loocv'``: Leave-One-Out CV proxy that exploits the Ridge hat-matrix
                   diagonal for efficiency.

    The estimator is fully compatible with the scikit-learn API and can be
    used inside ``Pipeline``, ``GridSearchCV``, and ``cross_val_score``.

    Parameters
    ----------
    tuning : {'bic', 'loocv'}, default='bic'
        Criterion used to jointly select ``(h_n, a_n)``.
    h_n_grid : array-like of shape (n_h,) or None, default=None
        Grid of positive Ridge regularisation values.  If ``None``, defaults
        to ``numpy.logspace(-2, 2, 20)``.
    a_n_grid : array-like of shape (n_a,) or None, default=None
        Grid of non-negative hard-threshold values.  If ``None``, defaults
        to ``numpy.linspace(0.01, 0.6, 15)``.
    fit_intercept : bool, default=True
        Whether to fit an intercept.  When ``True``, ``X`` and ``y`` are
        centred prior to estimation and the intercept is recovered
        analytically.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Sparse coefficient vector after thresholding.
    intercept_ : float
        Fitted intercept.  Zero when ``fit_intercept=False``.
    support_ : ndarray of shape (n_features,) of dtype bool
        Boolean mask of selected (non-zero) features.
    best_h_n_ : float
        Ridge regularisation parameter chosen by the tuning criterion.
    best_a_n_ : float
        Hard-threshold value chosen by the tuning criterion.
    n_features_in_ : int
        Number of features seen during ``fit``.

    References
    ----------
    Threshold Ridge regression is studied in:

        Lawal, J. (2026). Thresholded Ridge Regression for High-Dimensional
        Variable Selection. MPhil Dissertation, University of Cambridge.

    The Leave-One-Out CV proxy for Ridge is derived from:

        Hoerl, A. E. and Kennard, R. W. (1970). Ridge Regression: Biased
        Estimation for Nonorthogonal Problems. Technometrics, 12(1), 55-67.

    The BIC-based model selection criterion follows:

        Schwarz, G. (1978). Estimating the Dimension of a Model. Annals of
        Statistics, 6(2), 461-464.

    Examples
    --------
    >>> import numpy as np
    >>> from threshold_ridge import ThresholdedRidge
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 100))
    >>> beta = np.zeros(100)
    >>> beta[:10] = 1.0
    >>> y = X @ beta + rng.standard_normal(200)
    >>> tr = ThresholdedRidge(tuning='bic')
    >>> tr.fit(X, y)
    ThresholdedRidge()
    >>> tr.support_.sum()  # number of selected features
    10
    >>> tr.score(X, y)     # R-squared on training data
    0.9...
    """

    def __init__(
        self,
        tuning="bic",
        h_n_grid=None,
        a_n_grid=None,
        fit_intercept=True,
    ):
        self.tuning = tuning
        self.h_n_grid = h_n_grid
        self.a_n_grid = a_n_grid
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit the Thresholded Ridge model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training design matrix.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : ThresholdedRidge
            Fitted estimator.

        Raises
        ------
        ValueError
            If ``tuning`` is not one of ``{'bic', 'loocv'}``.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        if self.tuning not in _TUNING_STRATEGIES:
            raise ValueError(
                f"tuning must be one of {list(_TUNING_STRATEGIES)}, "
                f"got '{self.tuning}'."
            )

        h_n_grid = (
            np.asarray(self.h_n_grid, dtype=float)
            if self.h_n_grid is not None
            else np.logspace(-2, 2, 20)
        )
        a_n_grid = (
            np.asarray(self.a_n_grid, dtype=float)
            if self.a_n_grid is not None
            else np.linspace(0.01, 0.6, 15)
        )

        if self.fit_intercept:
            self._X_offset = X.mean(axis=0)
            self._y_offset = float(y.mean())
            X_fit = X - self._X_offset
            y_fit = y - self._y_offset
        else:
            self._X_offset = np.zeros(X.shape[1])
            self._y_offset = 0.0
            X_fit = X
            y_fit = y

        tuning_fn = _TUNING_STRATEGIES[self.tuning]
        coef, self.best_h_n_, self.best_a_n_ = tuning_fn(
            X_fit, y_fit, h_n_grid, a_n_grid
        )

        self.coef_ = coef
        self.support_ = coef != 0
        self.intercept_ = (
            self._y_offset - float(self._X_offset @ self.coef_)
            if self.fit_intercept
            else 0.0
        )

        return self

    def predict(self, X):
        """
        Predict target values for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self)
        X = check_array(X)
        return X @ self.coef_ + self.intercept_
