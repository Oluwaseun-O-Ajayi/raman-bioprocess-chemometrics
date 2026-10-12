"""Raman-specific preprocessing.

Cosmic rays hit the detector at random and leave narrow, very tall spikes, often
one or two pixels wide. They must be removed before smoothing or derivatives,
which would otherwise spread a spike into neighbouring wavenumbers.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def modified_z(x):
    """Modified z-score (Iglewicz and Hoaglin), based on the median, not the mean."""
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    if mad == 0:
        return np.zeros_like(x, dtype=float)
    return 0.6745 * (x - median) / mad


class Despike(BaseEstimator, TransformerMixin):
    """Remove cosmic-ray spikes.

    Each spectrum is compared with two local references over ``2 * window + 1``
    pixels: a quadratic fit (Savitzky-Golay) and a running median. A spike one or
    two pixels wide sits far above both. The top of a Raman band sits above the
    running median but on the quadratic fit, and the edge of a step sits on the
    running median, so neither is flagged. A point is a spike when it lies above
    both references by more than ``threshold`` (modified z-score of the
    residual); only positive deviations count, since cosmic rays add signal. Flagged points are replaced by linear
    interpolation between the nearest unflagged points. The outlier test follows
    Whitaker and Hayes (2018). The step acts on each spectrum separately, so
    fitting learns nothing.
    """

    def __init__(self, threshold=10.0, window=2):
        self.threshold = threshold
        self.window = window

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def spike_mask(self, X):
        from scipy.ndimage import median_filter
        from scipy.signal import savgol_filter
        X = np.asarray(X, dtype=float)
        width = 2 * self.window + 1
        r_fit = X - savgol_filter(X, width, 2, axis=1, mode="interp")
        r_med = X - median_filter(X, size=(1, width), mode="nearest")
        z_fit = np.vstack([modified_z(r) for r in r_fit])
        z_med = np.vstack([modified_z(r) for r in r_med])
        return (z_fit > self.threshold) & (z_med > self.threshold)

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        mask = self.spike_mask(X)
        idx = np.arange(X.shape[1])
        for i in np.flatnonzero(mask.any(axis=1)):
            good = ~mask[i]
            X[i, ~good] = np.interp(idx[~good], idx[good], X[i, good])
        return X
