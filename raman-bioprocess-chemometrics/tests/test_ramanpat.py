"""Fast checks on synthetic spectra. No data download is needed."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ramanpat import data
from ramanpat.lifecycle import correlated_subset, update_experiment
from ramanpat.models import cls_signatures, pls, pls_path, rmse, select_pls
from ramanpat.monitoring import build_monitor, q_residuals
from ramanpat.preprocessing import make_pipelines, output_axis
from ramanpat.raman import Despike
from ramanpat.uncertainty import conformal_halfwidth, coverage


def synthetic(n=60, p=300, seed=0):
    """Three Gaussian bands on a sloping background: two analytes and an interferent."""
    rng = np.random.default_rng(seed)
    w = np.linspace(400, 1800, p)
    band = lambda c, s: np.exp(-((w - c) / s) ** 2)  # noqa: E731
    S = np.vstack([band(1125, 15), band(928, 12), band(1450, 30)])
    C = rng.uniform(0, 10, (n, 3))
    X = 2000 + 0.5 * (1800 - w) + C @ (200 * S) + rng.normal(0, 5, (n, p))
    return w, X, C, 200 * S


# --- data -------------------------------------------------------------------

def test_valid_spectra_removes_flat_nan_and_saturated():
    X = np.random.default_rng(1).normal(100, 5, size=(4, 10))
    X[1] = 1.0
    X[2, 3] = np.nan
    X[3, 5] = data.SATURATION
    assert data.valid_spectra(X).tolist() == [True, False, False, False]


def test_replicates_share_a_group_and_never_cross_splits():
    Y = pd.DataFrame({"a": [1, 1, 2, 2, 3, 3, np.nan, np.nan], "b": [0, 0, 5, 5, 1, 1, 2, 2]})
    groups = data.sample_groups(Y)
    assert len(np.unique(groups)) == 4
    assert groups[0] == groups[1] and groups[6] == groups[7]
    labels = data.group_split(np.repeat(np.arange(100), 2), seed=3)
    for g in range(100):
        assert len(set(labels[2 * g:2 * g + 2])) == 1


# --- preprocessing and models ---------------------------------------------

def test_pipelines_finite_and_axis_matches():
    w, X, _, _ = synthetic()
    for name, pipe in make_pipelines(w).items():
        Z = pipe.fit(X).transform(X)
        assert np.isfinite(Z).all(), name
        assert Z.shape[1] == len(output_axis(pipe, w)), name


def test_pls_path_matches_refitting():
    w, X, C, _ = synthetic()
    full = pls(6).fit(X, C[:, 0])
    path = pls_path(full, X[:5])
    for a in (1, 3, 6):
        assert np.allclose(path[:, a - 1], np.ravel(pls(a).fit(X, C[:, 0]).predict(X[:5])))


def test_select_pls_recovers_signal():
    w, X, C, _ = synthetic(n=120)
    curve, a = select_pls(X[:80], C[:80, 0], X[80:], C[80:, 0], max_components=8)
    assert curve[a] < 0.3


def test_cls_signatures_recover_bands():
    w, X, C, S = synthetic(n=200)
    est = cls_signatures(X, C)
    for j in range(3):
        assert np.corrcoef(est[j], S[j])[0, 1] > 0.99


# --- Raman preprocessing ---------------------------------------------------

def test_despike_removes_spikes_and_keeps_bands():
    # 1,200 points over 400-1800 cm^-1 gives about 1.2 cm^-1 per point, as in the real data
    w, X, _, _ = synthetic(n=5, p=1200)
    clean = X.copy()
    X = X.copy()
    X[0, 100] += 3000
    X[1, 800:802] += 2000
    out = Despike().fit(X).transform(X)
    assert abs(out[0, 100] - clean[0, 100]) < 30
    assert np.all(np.abs(out[1, 800:802] - clean[1, 800:802]) < 30)
    keep = np.ones_like(X, dtype=bool)
    keep[0, 100] = False
    keep[1, 800:802] = False
    assert np.max(np.abs(out - clean)[keep]) < 0.01 * clean.max()


def test_despike_ignores_a_step():
    x = np.r_[np.zeros(100), np.ones(100) * 50] + np.random.default_rng(0).normal(0, 0.1, 200)
    assert np.allclose(Despike().fit(x[None]).transform(x[None]), x[None])


# --- monitoring and intervals ----------------------------------------------

def test_t2_limit_matches_formula():
    w, X, C, _ = synthetic(n=50)
    model = pls(3).fit(X, C[:, 0])
    n, a = 50, 3
    expected = a * (n ** 2 - 1) / (n * (n - a)) * stats.f.ppf(0.99, a, n - a)
    assert build_monitor(model, X).t2_limit(0.99) == pytest.approx(expected)


def test_q_flags_a_new_band():
    w, X, C, _ = synthetic(n=80)
    model = pls(3).fit(X, C[:, 0])
    mon = build_monitor(model, X)
    odd = X[:5] + 500 * np.exp(-((w - 700) / 20) ** 2)
    assert (q_residuals(model, odd) > mon.q_limit(0.99)).all()


def test_conformal_interval_covers_at_the_stated_rate():
    rng = np.random.default_rng(0)
    y_val, y_new = rng.normal(size=2000), rng.normal(size=5000)
    h = conformal_halfwidth(y_val, np.zeros(2000), 0.95)
    assert 0.93 < coverage(y_new, np.zeros(5000), h) < 0.97


# --- lifecycle --------------------------------------------------------------

def test_correlated_subset_reaches_target_correlation():
    rng = np.random.default_rng(0)
    y1, y2 = rng.uniform(0, 20, 2000), rng.uniform(0, 2, 2000)
    groups = np.arange(2000)
    chosen = correlated_subset(y1, y2, groups, 300, 0.9, rng)
    assert len(chosen) == 300
    assert np.corrcoef(y1[chosen], y2[chosen])[0, 1] >= 0.9


def test_update_experiment_runs_and_improves_on_shifted_process():
    w, Xc, Cc, _ = synthetic(n=150, seed=1)
    _, Xp, Cp, _ = synthetic(n=60, seed=2)
    Xp = Xp + 300 * np.exp(-((w - 1110) / 20) ** 2)   # matrix band overlapping the analyte band
    groups = np.repeat(np.arange(30), 2)
    pool, held = groups < 15, groups >= 15
    rows = pd.DataFrame(update_experiment(Xc, Cc[:, 0], Xp, Cp[:, 0], groups, pool, held, 3,
                                          n_added=(0, 10), n_draws=3))
    base = rows.loc[rows["policy"] == "none", "rmsep"].iloc[0]
    weighted = rows.loc[rows["policy"] == "augment_weighted", "rmsep"].mean()
    assert weighted < base


# --- deployment -------------------------------------------------------------

def test_deploy_round_trip_and_axis_check(tmp_path):
    from ramanpat import deploy
    w, X, C, _ = synthetic()
    prep = make_pipelines(w)["region_snv"].fit(X)
    Xp = prep.transform(X)
    model = pls(3).fit(Xp, C[:, 0])
    bundle = deploy.make_bundle(prep, model, build_monitor(model, Xp), w, C[:, 0], 0.5, "region_snv", "Glucose", 1.0)
    deploy.save(bundle, tmp_path / "m.joblib")
    out = deploy.predict(deploy.load(tmp_path / "m.joblib"), X[:3], w)
    assert np.allclose(out["prediction"], np.ravel(model.predict(Xp[:3])))
    assert np.allclose(out["interval95_high"] - out["interval95_low"], 2.0)
    with pytest.raises(ValueError):
        deploy.predict(bundle, X[:3], w + 1.0)
    assert rmse([1, 2], [1, 2]) == 0
