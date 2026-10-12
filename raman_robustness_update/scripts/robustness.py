"""Repeat the headline results over several splits of the designed data.

    python scripts/robustness.py

Writes results/robustness.csv and results/robustness_summary.json. For each
seed, the designed data are re-split by sample, preprocessing and latent
variables are re-selected, and the final models are re-scored on the designed
test set and the E. coli samples. The weighted 10-sample update is repeated with
5 draws per seed. A bootstrap over E. coli test samples gives confidence
intervals for the source-study comparison.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.base import clone  # noqa: E402

from ramanpat import data  # noqa: E402
from ramanpat.lifecycle import update_experiment  # noqa: E402
from ramanpat.models import pls, rmse, select_pls  # noqa: E402
from ramanpat.monitoring import build_monitor, q_residuals  # noqa: E402
from ramanpat.preprocessing import make_pipelines  # noqa: E402

warnings.filterwarnings("ignore")
ANALYTES = ("Glucose", "Acetate")
SEEDS = (0, 1, 2, 3, 4)


def r2(y, p):
    p = np.ravel(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def main():
    sm, ec = data.load("substratemix", ROOT / "data"), data.load("ecoli", ROOT / "data")
    valid = data.valid_spectra(sm.X)
    pipelines = make_pipelines(sm.wavenumbers)
    rows = []
    for seed in SEEDS:
        split = data.group_split(sm.group, seed=seed)
        train_rows = valid & (split == "train")
        Xs, Xe = {}, {}
        for name, p in pipelines.items():
            f = clone(p).fit(sm.X[train_rows])
            Xs[name], Xe[name] = f.transform(sm.X), f.transform(ec.X)
        for analyte in ANALYTES:
            y, ye = sm.analyte(analyte), ec.analyte(analyte)
            m = valid & ~np.isnan(y)
            tr, va, te = (m & (split == s) for s in ("train", "val", "test"))
            me = ~np.isnan(ye)
            best = None
            for name in pipelines:
                curve, a = select_pls(Xs[name][tr], y[tr], Xs[name][va], y[va])
                if best is None or curve[a] < best[2]:
                    best = (name, a, curve[a])
            name, a, _ = best
            model = pls(a).fit(Xs[name][tr], y[tr])
            p_te, p_ec = np.ravel(model.predict(Xs[name][te])), np.ravel(model.predict(Xe[name][me]))
            mon = build_monitor(model, Xs[name][tr])
            flag = (mon.t2(model.transform(Xe[name][me])) > mon.t2_limit(0.99)) | \
                   (q_residuals(model, Xe[name][me]) > mon.q_limit(0.99))
            pool, held = me & (ec.source_split == "train"), me & (ec.source_split != "train")
            upd = pd.DataFrame(update_experiment(Xs[name][tr], y[tr], Xe[name], np.nan_to_num(ye), ec.group,
                                                 pool, held, a, n_added=(0, 10), n_draws=5, seed=seed))
            w10 = upd[(upd["policy"] == "augment_weighted") & (upd["samples_added"] == 10)]["rmsep"].mean()
            base = upd[upd["policy"] == "none"]["rmsep"].iloc[0]
            yf, pf = ye[me], p_ec
            rows.append({"seed": seed, "analyte": analyte, "pipeline": name, "latent_variables": a,
                         "rmsep_designed_test": rmse(y[te], p_te), "r2_designed_test": r2(y[te], p_te),
                         "rmsep_ecoli": rmse(yf, pf), "bias_ecoli": float(np.mean(pf - yf)), "r2_ecoli": r2(yf, pf),
                         "flagged99_ecoli": float(flag.mean()),
                         "rmsep_ecoli_flagged": rmse(yf[flag], pf[flag]) if flag.any() else np.nan,
                         "rmsep_ecoli_not_flagged": rmse(yf[~flag], pf[~flag]) if (~flag).any() else np.nan,
                         "update_heldout_none": float(base), "update_heldout_weighted10": float(w10)})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "robustness.csv", index=False)
    num = df.drop(columns=["pipeline"]).groupby("analyte")
    summary = {"seeds": list(SEEDS), "mean": num.mean().drop(columns="seed").round(3).to_dict(orient="index"),
               "sd": num.std().drop(columns="seed").round(3).to_dict(orient="index"),
               "pipelines_chosen": df.groupby("analyte")["pipeline"].apply(list).to_dict()}

    # Bootstrap over E. coli test samples (source-study split): PLS trained on E. coli training cultivations.
    rng = np.random.default_rng(0)
    split = data.group_split(sm.group, seed=0)
    boot = {}
    for analyte, name in (("Glucose", "full_snv"), ("Acetate", "full_raw")):
        f = clone(pipelines[name]).fit(sm.X[valid & (split == "train")])
        Xe = f.transform(ec.X)
        ye = ec.analyte(analyte)
        me = ~np.isnan(ye)
        etr, eva, ete = (me & (ec.source_split == s) for s in ("train", "val", "test"))
        _, k = select_pls(Xe[etr], ye[etr], Xe[eva], ye[eva], max_components=20)
        err = np.abs(ye[ete] - np.ravel(pls(k).fit(Xe[etr], ye[etr]).predict(Xe[ete])))
        groups = ec.group[ete]
        ug = np.unique(groups)
        maes = []
        for _ in range(2000):
            pick = rng.choice(ug, size=len(ug), replace=True)
            idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
            maes.append(err[idx].mean())
        boot[analyte] = {"mae": round(float(err.mean()), 3),
                         "ci95": [round(float(np.percentile(maes, 2.5)), 3), round(float(np.percentile(maes, 97.5)), 3)],
                         "latent_variables": int(k), "test_samples": int(len(ug))}
    summary["ecoli_train_only_mae_bootstrap"] = boot
    (ROOT / "results" / "robustness_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
