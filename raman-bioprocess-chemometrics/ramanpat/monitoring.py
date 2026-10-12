"""Multivariate monitoring of new spectra against a PLS calibration.

Two statistics, both computed in the preprocessed spectral space:

Hotelling's T2 measures how far a spectrum's scores lie from the centre of the
calibration data, inside the PLS model plane. A high T2 means an unusual
combination of the variation the model already knows, for example a
concentration outside the calibrated range.

Q (squared prediction error) is the part of a spectrum the PLS model cannot
reconstruct. A high Q means variation the model has never seen, for example a
new compound, a fluorescence background, or an instrument change.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Monitor:
    score_var: np.ndarray   # variance of each latent variable in the calibration set
    n_cal: int
    n_components: int
    q_cal: np.ndarray       # Q values of the calibration spectra

    def t2(self, scores):
        return np.sum(scores ** 2 / self.score_var, axis=1)

    def t2_limit(self, confidence=0.95):
        """Control limit for T2 of new observations (F-distribution form)."""
        n, a = self.n_cal, self.n_components
        f = stats.f.ppf(confidence, a, n - a)
        return a * (n ** 2 - 1) / (n * (n - a)) * f

    def q_limit(self, confidence=0.95):
        """Control limit for Q using Box's scaled chi-squared approximation.

        Q is approximated as g * chi2(h), with g and h matched to the mean and
        variance of the calibration Q values.
        """
        mean, var = float(np.mean(self.q_cal)), float(np.var(self.q_cal, ddof=1))
        g, h = var / (2 * mean), 2 * mean ** 2 / var
        return g * stats.chi2.ppf(confidence, h)


def q_residuals(pls_model, X_preprocessed):
    """Squared reconstruction error of each spectrum in the fitted PLS model."""
    scores = pls_model.transform(X_preprocessed)
    reconstructed = pls_model.inverse_transform(scores)
    return np.sum((X_preprocessed - reconstructed) ** 2, axis=1)


def build_monitor(pls_model, X_cal_preprocessed):
    scores = pls_model.transform(X_cal_preprocessed)
    return Monitor(
        score_var=np.var(scores, axis=0, ddof=1),
        n_cal=X_cal_preprocessed.shape[0],
        n_components=scores.shape[1],
        q_cal=q_residuals(pls_model, X_cal_preprocessed),
    )
