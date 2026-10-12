"""Prediction intervals by split conformal prediction.

The absolute errors of the model on a validation set it was not fitted on give
an empirical distribution of prediction error. The interval for a new spectrum
is the prediction plus or minus the appropriate quantile of those errors. If new
spectra resemble the validation spectra, the interval covers the true value at
the stated rate; if they do not, coverage drops, and measuring that drop is part
of the test.
"""

import numpy as np


def conformal_halfwidth(y_val, pred_val, coverage=0.95):
    """Half-width of the interval, with the finite-sample correction."""
    errors = np.sort(np.abs(np.asarray(y_val) - np.ravel(pred_val)))
    n = len(errors)
    k = int(np.ceil((n + 1) * coverage)) - 1
    return float(errors[min(k, n - 1)])


def coverage(y_true, pred, halfwidth):
    return float(np.mean(np.abs(np.asarray(y_true) - np.ravel(pred)) <= halfwidth))
