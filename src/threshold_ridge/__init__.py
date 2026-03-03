"""
threshold-ridge
===============
Scikit-learn compatible Thresholded Ridge regression for high-dimensional
variable selection.

Classes
-------
ThresholdedRidge
    Two-step estimator: Ridge regression followed by hard thresholding,
    with hyperparameters tuned by BIC or leave-one-out cross-validation.
"""

from .estimators import ThresholdedRidge

__version__ = "0.1.0"
__all__ = ["ThresholdedRidge"]
