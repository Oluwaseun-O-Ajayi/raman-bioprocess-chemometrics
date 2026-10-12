"""Experiments on calibration design and model maintenance.

All functions take spectra that are already preprocessed, so that the
preprocessing fitted on the calibration set is held fixed, as it would be for a
model in service.
"""

import numpy as np
from sklearn.decomposition import PCA

from ramanpat.models import pls, pls_path, rmse
from ramanpat.monitoring import build_monitor, q_residuals


def _choose_lv_by_group_cv(X, y, groups, max_components):
    """Leave-one-group-out choice of latent variables for a small data set."""
    unique = np.unique(groups)
    max_components = max(1, min(max_components, len(unique) - 1, X.shape[1]))
    if len(unique) < 3:
        return 1
    pred = np.zeros((len(y), max_components))
    for g in unique:
        test = groups == g
        if np.ptp(y[~test]) == 0:          # constant reference values: nothing to model
            pred[test] = y[~test][0]
            continue
        model = pls(max_components).fit(X[~test], y[~test])
        pred[test] = pls_path(model, X[test])
    errors = [rmse(y, pred[:, a]) for a in range(max_components)]
    return int(np.argmin(errors)) + 1


def update_experiment(Xc, yc, Xp, yp, groups_p, pool, held_out, n_lv,
                      n_added=(0, 2, 5, 10, 20, 40, 70), n_draws=20, weight=10, seed=0):
    """Add ``k`` process samples to the calibration and score on held-out process samples.

    Policies, for each k and each random draw of k samples from the pool:
    - augment: calibration spectra plus the k process samples, same number of LVs
    - augment_weighted: as augment, with each process spectrum repeated ``weight``
      times so that a few samples can shift a model built on thousands
    - local: a model built on the k process samples alone, LVs chosen by
      leave-one-sample-out cross-validation

    Values of k above the pool size are capped at the pool size, which needs a
    single draw. Process samples are drawn whole (both replicate spectra). Monitoring limits
    are rebuilt from each training set and applied to the held-out spectra.
    """
    rng = np.random.default_rng(seed)
    pool_groups = np.unique(groups_p[pool])
    Xh, yh = Xp[held_out], yp[held_out]
    rows = []
    for k in sorted({min(k, len(pool_groups)) for k in n_added}):
        draws = 1 if k in (0, len(pool_groups)) else n_draws
        for d in range(draws):
            chosen = rng.choice(pool_groups, size=k, replace=False) if k else np.array([], dtype=int)
            idx = np.flatnonzero(pool & np.isin(groups_p, chosen))
            policies = ["augment"] if k == 0 else ["augment", "augment_weighted", "local"]
            for policy in policies:
                if policy == "local":
                    if k < 3:
                        continue
                    a = _choose_lv_by_group_cv(Xp[idx], yp[idx], groups_p[idx], 10)
                    Xt, yt = Xp[idx], yp[idx]
                else:
                    rep = weight if policy == "augment_weighted" else 1
                    Xt = np.vstack([Xc] + [Xp[idx]] * rep)
                    yt = np.concatenate([yc] + [yp[idx]] * rep)
                    a = n_lv
                if np.ptp(yt) == 0:            # all drawn samples share one value
                    rows.append({"policy": policy, "samples_added": k, "draw": d, "latent_variables": 0,
                                 "rmsep": rmse(yh, np.full(len(yh), yt[0])),
                                 "bias": float(yt[0] - yh.mean()), "fraction_flagged_99": np.nan})
                    continue
                model = pls(a).fit(Xt, yt)
                pred = np.ravel(model.predict(Xh))
                monitor = build_monitor(model, Xt)
                flagged = (monitor.t2(model.transform(Xh)) > monitor.t2_limit(0.99)) | \
                          (q_residuals(model, Xh) > monitor.q_limit(0.99))
                rows.append({"policy": "none" if k == 0 else policy, "samples_added": k, "draw": d,
                             "latent_variables": a, "rmsep": rmse(yh, pred),
                             "bias": float(np.mean(pred - yh)),
                             "fraction_flagged_99": float(flagged.mean())})
    return rows


def correlated_subset(y1, y2, groups, n_groups, rho, rng):
    """Pick ``n_groups`` samples whose two analytes are strongly correlated.

    Both analytes are scaled to 0-1. Samples near the diagonal (both low or both
    high) are drawn with higher probability, and the band is narrowed until the
    correlation of the chosen samples reaches ``rho``.
    """
    unique, first = np.unique(groups, return_index=True)
    u = (y1[first] - y1[first].min()) / np.ptp(y1[first])
    v = (y2[first] - y2[first].min()) / np.ptp(y2[first])
    distance = np.abs(u - v)
    for width in np.linspace(0.5, 0.01, 50):
        candidates = np.flatnonzero(distance <= width)
        if len(candidates) < n_groups:
            break
        pick = rng.choice(candidates, size=n_groups, replace=False)
        if np.corrcoef(u[pick], v[pick])[0, 1] >= rho:
            return unique[pick]
    raise ValueError("could not reach the requested correlation")


def design_experiment(X, Y1, Y2, groups, X_test, y1_test, y2_test, n_lv1, n_lv2,
                      n_groups=400, rho=0.9, n_draws=20, seed=0):
    """Correlated versus independent calibration design, same number of samples.

    For each draw, one calibration subset is taken at random from the designed
    data (analytes nearly uncorrelated) and one is chosen so that the two
    analytes are correlated at ``rho``, as glucose and its products are in a
    real fermentation. Both are scored on the independent test set, where the
    correlation is absent.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rows = []
    for d in range(n_draws):
        subsets = {
            "independent": rng.choice(unique, size=n_groups, replace=False),
            "correlated": correlated_subset(Y1, Y2, groups, n_groups, rho, rng),
        }
        for design, members in subsets.items():
            mask = np.isin(groups, members)
            r = float(np.corrcoef(Y1[mask], Y2[mask])[0, 1])
            for analyte, y, y_test, a in [("Glucose", Y1, y1_test, n_lv1), ("Acetate", Y2, y2_test, n_lv2)]:
                model = pls(a).fit(X[mask], y[mask])
                rows.append({"design": design, "draw": d, "analyte": analyte,
                             "training_correlation": r,
                             "rmsep_test": rmse(y_test, model.predict(X_test))})
    return rows


def pca_scores(X_cal, X_new, n_components=2):
    """PCA fitted on calibration and new spectra together (a diagnostic view)."""
    pca = PCA(n_components=n_components).fit(np.vstack([X_cal, X_new]))
    return pca.transform(X_cal), pca.transform(X_new), pca.explained_variance_ratio_
