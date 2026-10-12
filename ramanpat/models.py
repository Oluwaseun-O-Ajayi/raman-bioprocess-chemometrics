"""PLS and ridge models with selection on a validation set."""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def pls(n_components):
    # scale=False: spectra share a common scale after preprocessing, so only
    # mean-centring is applied (the usual choice for spectroscopy).
    return PLSRegression(n_components=n_components, scale=False)


def ridge(alpha):
    return Pipeline([("centre", StandardScaler(with_std=False)), ("ridge", Ridge(alpha=alpha))])


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.ravel(y_pred)) ** 2)))


def pls_path(model, X):
    """Predictions for 1..A latent variables from one fitted PLS1 model.

    The coefficients for the first ``a`` components are
    W_a (P_a' W_a)^-1 q_a', so one fit with A components gives all smaller
    models without refitting.
    """
    W, P = model.x_weights_, model.x_loadings_
    q = np.ravel(model.y_loadings_)
    Xc = np.asarray(X) - model._x_mean       # scale=False, so no division by a std
    y_mean = float(np.ravel(model._y_mean)[0])
    out = []
    for a in range(1, W.shape[1] + 1):
        try:
            b = W[:, :a] @ np.linalg.solve(P[:, :a].T @ W[:, :a], q[:a])
        except np.linalg.LinAlgError:      # y fully explained by fewer components
            b = W[:, :a] @ np.linalg.lstsq(P[:, :a].T @ W[:, :a], q[:a], rcond=None)[0]
        out.append(Xc @ b + y_mean)
    return np.column_stack(out)


def select_pls(X_train, y_train, X_val, y_val, max_components=25):
    """Validation RMSE for 1..max_components; choose the smallest within 2% of the minimum."""
    full = pls(max_components).fit(X_train, y_train)
    pred = pls_path(full, X_val)
    curve = {a + 1: rmse(y_val, pred[:, a]) for a in range(pred.shape[1])}
    best = min(curve.values())
    chosen = min(a for a, v in curve.items() if v <= 1.02 * best)
    return curve, chosen


def select_ridge(X_train, y_train, X_val, y_val, alphas=(1e-2, 1e-1, 1, 10, 100, 1e3, 1e4)):
    curve = {a: rmse(y_val, ridge(a).fit(X_train, y_train).predict(X_val)) for a in alphas}
    return curve, min(curve, key=curve.get)


def cls_signatures(X, Y):
    """Spectral signature of each analyte by classical least squares.

    Fits X = 1 b0 + Y S by least squares, using only samples with every analyte
    measured. Row j of S is the change in the spectrum per g/L of analyte j,
    holding the others fixed. This works because the designed concentrations
    are close to independent.
    """
    Y = np.asarray(Y, dtype=float)
    ok = ~np.isnan(Y).any(axis=1)
    A = np.column_stack([np.ones(ok.sum()), Y[ok]])
    coef = np.linalg.lstsq(A, np.asarray(X)[ok], rcond=None)[0]
    return coef[1:]
