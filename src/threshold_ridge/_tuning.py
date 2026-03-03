"""
Internal tuning routines for the Thresholded Ridge estimator.

Both functions assume the design matrix X and response y have already been
centred (intercept removed).  Centring is handled by the public estimator
class ThresholdedRidge.
"""

import numpy as np
from sklearn.linear_model import Ridge


def tune_loocv(X, y, h_n_grid, a_n_grid):
    """
    Tune Thresholded Ridge hyperparameters via a Leave-One-Out CV proxy.

    The LOO CV score is computed analytically from the Ridge hat-matrix
    diagonal, avoiding the need to refit the model for each fold.  For a
    given (h_n, a_n) pair the score is:

        LOO-CV = mean( (r_i / (1 - w_i))^2 )

    where r_i are residuals from the thresholded fit and w_i are the
    diagonal entries of the Ridge hat matrix H = X(X'X + h_n I)^{-1} X'.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Centred design matrix (no intercept column).
    y : ndarray of shape (n_samples,)
        Centred response vector.
    h_n_grid : ndarray of shape (n_h,)
        Grid of Ridge regularisation parameters to search over.
    a_n_grid : ndarray of shape (n_a,)
        Grid of hard-threshold values to search over.

    Returns
    -------
    best_coef : ndarray of shape (n_features,)
        Thresholded coefficient vector at the optimal (h_n, a_n).
    best_h_n : float
        Optimal Ridge regularisation parameter.
    best_a_n : float
        Optimal threshold value.
    """
    n_samples, n_features = X.shape
    min_cv_score = np.inf
    best_h_n, best_a_n = h_n_grid[0], a_n_grid[0]
    best_coef = np.zeros(n_features)

    ridge_fits = {}
    hat_diagonals = {}

    for h_n in h_n_grid:
        try:
            ridge = Ridge(alpha=h_n, fit_intercept=False, solver="svd").fit(X, y)
            ridge_fits[h_n] = ridge.coef_
            inv_term = np.linalg.inv(X.T @ X + h_n * np.eye(n_features))
            w_vector = np.sum(X * (X @ inv_term), axis=1)
            if np.any(w_vector >= 1.0 - 1e-9) or np.any(w_vector < 0):
                hat_diagonals[h_n] = None
            else:
                hat_diagonals[h_n] = w_vector
        except np.linalg.LinAlgError:
            ridge_fits[h_n] = None
            hat_diagonals[h_n] = None

    for h_n in h_n_grid:
        theta_hat = ridge_fits.get(h_n)
        w_vector = hat_diagonals.get(h_n)
        if theta_hat is None or w_vector is None:
            continue
        for a_n in a_n_grid:
            theta_tilde = theta_hat * (np.abs(theta_hat) > a_n)
            residuals = y - X @ theta_tilde
            denominators = np.maximum(1.0 - w_vector, 1e-9)
            cv_score = np.mean((residuals / denominators) ** 2)
            if cv_score < min_cv_score:
                min_cv_score = cv_score
                best_h_n = h_n
                best_a_n = a_n
                best_coef = theta_tilde.copy()

    return best_coef, best_h_n, best_a_n


def tune_bic(X, y, h_n_grid, a_n_grid):
    """
    Tune Thresholded Ridge hyperparameters using the Bayesian Information
    Criterion (BIC).

    For a given (h_n, a_n) pair the BIC is:

        BIC = n * log(RSS / n) + k * log(n)

    where RSS is the residual sum of squares from the thresholded fit and
    k is the number of selected variables plus one (for sigma).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Centred design matrix (no intercept column).
    y : ndarray of shape (n_samples,)
        Centred response vector.
    h_n_grid : ndarray of shape (n_h,)
        Grid of Ridge regularisation parameters to search over.
    a_n_grid : ndarray of shape (n_a,)
        Grid of hard-threshold values to search over.

    Returns
    -------
    best_coef : ndarray of shape (n_features,)
        Thresholded coefficient vector at the optimal (h_n, a_n).
    best_h_n : float
        Optimal Ridge regularisation parameter.
    best_a_n : float
        Optimal threshold value.
    """
    n_samples, n_features = X.shape
    min_bic = np.inf
    best_h_n, best_a_n = h_n_grid[0], a_n_grid[0]
    best_coef = np.zeros(n_features)

    ridge_coefs = {}
    for h_n in h_n_grid:
        try:
            ridge = Ridge(alpha=h_n, fit_intercept=False, solver="svd").fit(X, y)
            ridge_coefs[h_n] = ridge.coef_
        except Exception:
            ridge_coefs[h_n] = None

    for h_n in h_n_grid:
        theta_hat = ridge_coefs.get(h_n)
        if theta_hat is None:
            continue
        for a_n in a_n_grid:
            beta_tr = theta_hat * (np.abs(theta_hat) > a_n)
            rss = np.sum((y - X @ beta_tr) ** 2)
            k = int(np.sum(beta_tr != 0)) + 1
            bic = n_samples * np.log(max(rss, 1e-9) / n_samples) + k * np.log(n_samples)
            if bic < min_bic:
                min_bic = bic
                best_h_n = h_n
                best_a_n = a_n
                best_coef = beta_tr.copy()

    return best_coef, best_h_n, best_a_n
