"""Full analysis. Writes tables to results/, figures to figures/, models to models/.

    python scripts/download_data.py     # once
    python scripts/run_analysis.py
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

from ramanpat import data, deploy  # noqa: E402
from ramanpat import figures as fig  # noqa: E402
from ramanpat.lifecycle import design_experiment, pca_scores, update_experiment  # noqa: E402
from ramanpat.models import (cls_signatures, pls, ridge, rmse,  # noqa: E402
                             select_pls, select_ridge)
from ramanpat.monitoring import build_monitor, q_residuals  # noqa: E402
from ramanpat.preprocessing import make_pipelines, output_axis  # noqa: E402
from ramanpat.raman import Despike  # noqa: E402
from ramanpat.uncertainty import conformal_halfwidth, coverage  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
ANALYTES = ("Glucose", "Acetate")
N_DRAWS = 20


def r2(y, p):
    y, p = np.asarray(y), np.ravel(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def main():
    res, figs, models_dir = ROOT / "results", ROOT / "figures", ROOT / "models"
    for d in (res, figs, models_dir):
        d.mkdir(exist_ok=True)

    sm, ec = data.load("substratemix", ROOT / "data"), data.load("ecoli", ROOT / "data")
    valid = data.valid_spectra(sm.X)
    split = data.group_split(sm.group, seed=0)
    train_rows = valid & (split == "train")
    summary = {
        "designed_spectra": int(len(sm.X)), "designed_samples": int(len(np.unique(sm.group))),
        "designed_spectra_removed_saturated": int((~valid).sum()),
        "designed_split_spectra": {s: int((valid & (split == s)).sum()) for s in ("train", "val", "test")},
        "ecoli_spectra": int(len(ec.X)), "ecoli_samples": int(len(np.unique(ec.group))),
        "ecoli_spectra_invalid": int((~data.valid_spectra(ec.X)).sum()),
    }

    # Preprocess everything once per pipeline, each fitted on calibration (train) spectra only.
    pipelines = make_pipelines(sm.wavenumbers)
    fitted, Xs, Xe = {}, {}, {}
    for name, p in pipelines.items():
        fitted[name] = clone(p).fit(sm.X[train_rows])
        Xs[name], Xe[name] = fitted[name].transform(sm.X), fitted[name].transform(ec.X)

    # ------------------------------------------------------------------
    # 1. Model selection on the validation set.
    # ------------------------------------------------------------------
    sel_rows, curve_rows, chosen = [], [], {}
    masks = {}
    for analyte in ANALYTES:
        y, ye = sm.analyte(analyte), ec.analyte(analyte)
        m = valid & ~np.isnan(y)
        masks[analyte] = {s: m & (split == s) for s in ("train", "val", "test")}
        masks[analyte]["ecoli"] = ~np.isnan(ye)
        tr, va, te, me = (masks[analyte][k] for k in ("train", "val", "test", "ecoli"))
        for name in pipelines:
            curve, a = select_pls(Xs[name][tr], y[tr], Xs[name][va], y[va])
            curve_rows += [{"analyte": analyte, "pipeline": name, "latent_variables": k, "rmse_val": v}
                           for k, v in curve.items()]
            rcurve, alpha = select_ridge(Xs[name][tr], y[tr], Xs[name][va], y[va])
            for model_name, setting, val, est in [("PLS", a, curve[a], pls(a)),
                                                  ("Ridge", alpha, rcurve[alpha], ridge(alpha))]:
                est.fit(Xs[name][tr], y[tr])
                pe = np.ravel(est.predict(Xe[name][me]))
                sel_rows.append({"analyte": analyte, "pipeline": name, "model": model_name,
                                 "setting": setting, "rmse_val": round(val, 3),
                                 "rmsep_designed_test": round(rmse(y[te], est.predict(Xs[name][te])), 3),
                                 "rmsep_ecoli": round(rmse(ye[me], pe), 3),
                                 "bias_ecoli": round(float(np.mean(pe - ye[me])), 3)})
        pls_rows = [r for r in sel_rows if r["analyte"] == analyte and r["model"] == "PLS"]
        best = min(pls_rows, key=lambda r: r["rmse_val"])
        chosen[analyte] = (best["pipeline"], int(best["setting"]))
        print(analyte, "selected", chosen[analyte], flush=True)
    selection = pd.DataFrame(sel_rows)
    selection.to_csv(res / "model_selection.csv", index=False)
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(res / "pls_validation_curves.csv", index=False)

    # Rank agreement between validation error and E. coli error across PLS pipelines.
    from scipy.stats import spearmanr
    rank = {}
    for analyte in ANALYTES:
        p = selection[(selection["analyte"] == analyte) & (selection["model"] == "PLS")]
        rank[analyte] = round(float(spearmanr(p["rmse_val"], p["rmsep_ecoli"])[0]), 2)

    # ------------------------------------------------------------------
    # 2. Final models: performance, monitoring, intervals, deployment bundle.
    # ------------------------------------------------------------------
    pred_fig, mon_fig, coef_axes, coefs, final = {}, {}, {}, {}, {}
    for analyte in ANALYTES:
        name, a = chosen[analyte]
        y, ye = sm.analyte(analyte), ec.analyte(analyte)
        tr, va, te, me = (masks[analyte][k] for k in ("train", "val", "test", "ecoli"))
        model = pls(a).fit(Xs[name][tr], y[tr])
        p_va, p_te = np.ravel(model.predict(Xs[name][va])), np.ravel(model.predict(Xs[name][te]))
        p_ec = np.ravel(model.predict(Xe[name][me]))
        h = conformal_halfwidth(y[va], p_va)
        monitor = build_monitor(model, Xs[name][tr])
        t2l, ql = monitor.t2_limit(0.99), monitor.q_limit(0.99)
        stats = {}
        for label, X in [("cal", Xs[name][te]), ("proc", Xe[name][me])]:
            stats[f"{label}_t2"] = monitor.t2(model.transform(X))
            stats[f"{label}_q"] = q_residuals(model, X)
        flag_te = (stats["cal_t2"] > t2l) | (stats["cal_q"] > ql)
        flag_ec = (stats["proc_t2"] > t2l) | (stats["proc_q"] > ql)
        final[analyte] = {
            "pipeline": name, "latent_variables": a,
            "designed_range": [float(np.nanmin(y[tr])), float(np.nanmax(y[tr]))],
            "ecoli_range": [float(ye[me].min()), float(ye[me].max())],
            "ecoli_sd": round(float(ye[me].std()), 3),
            "rmse_val": round(rmse(y[va], p_va), 3),
            "rmsep_designed_test": round(rmse(y[te], p_te), 3),
            "r2_designed_test": round(r2(y[te], p_te), 3),
            "rmsep_ecoli": round(rmse(ye[me], p_ec), 3),
            "bias_ecoli": round(float(np.mean(p_ec - ye[me])), 3),
            "r2_ecoli": round(r2(ye[me], p_ec), 3),
            "interval95_halfwidth": round(h, 3),
            "coverage95_designed_test": round(coverage(y[te], p_te, h), 3),
            "coverage95_ecoli": round(coverage(ye[me], p_ec, h), 3),
            "t2_limit_99": round(float(t2l), 2), "q_limit_99": round(float(ql), 4),
            "flagged99_designed_test": round(float(flag_te.mean()), 3),
            "flagged99_ecoli": round(float(flag_ec.mean()), 3),
            "t2_above99_ecoli": round(float((stats["proc_t2"] > t2l).mean()), 3),
            "q_above99_ecoli": round(float((stats["proc_q"] > ql).mean()), 3),
            "rmsep_ecoli_flagged": round(rmse(ye[me][flag_ec], p_ec[flag_ec]), 3) if flag_ec.any() else None,
            "rmsep_ecoli_not_flagged": round(rmse(ye[me][~flag_ec], p_ec[~flag_ec]), 3) if (~flag_ec).any() else None,
        }
        pred_fig[analyte] = {"cal_ref": y[te], "cal_pred": p_te, "proc_ref": ye[me], "proc_pred": p_ec}
        mon_fig[analyte] = dict(stats, t2_limit=t2l, q_limit=ql)
        coef_axes[analyte] = output_axis(fitted[name], sm.wavenumbers)
        coefs[analyte] = np.ravel(model.coef_)
        pd.DataFrame({"reference": ye[me], "prediction": p_ec, "t2": stats["proc_t2"], "q": stats["proc_q"],
                      "flagged99": flag_ec}).to_csv(res / f"ecoli_predictions_{analyte.lower()}.csv", index=False)
        deploy.save(deploy.make_bundle(fitted[name], model, monitor, sm.wavenumbers, y[tr], rmse(y[va], p_va),
                                       name, analyte, h), models_dir / f"{analyte.lower()}_pls.joblib")
        print(analyte, final[analyte], flush=True)

    # Published split check: replicate spectra of one sample appear in both its train and test files.
    leak = {}
    for analyte in ANALYTES:
        name, a = chosen[analyte]
        y = sm.analyte(analyte)
        m = valid & ~np.isnan(y)
        ptr, pte = m & (sm.source_split == "train"), m & (sm.source_split == "test")
        shared = pte & np.isin(sm.group, sm.group[sm.source_split == "train"])
        p = np.ravel(pls(a).fit(Xs[name][ptr], y[ptr]).predict(Xs[name]))
        leak[analyte] = {"published_test_rmsep": round(rmse(y[pte], p[pte]), 3),
                         "published_test_spectra": int(pte.sum()),
                         "published_test_spectra_sharing_a_sample_with_train": int(shared.sum()),
                         "rmsep_sharing": round(rmse(y[shared], p[shared]), 3),
                         "rmsep_not_sharing": round(rmse(y[pte & ~shared], p[pte & ~shared]), 3)}

    # ------------------------------------------------------------------
    # 3. PCA and analyte signatures.
    # ------------------------------------------------------------------
    gname = chosen["Glucose"][0]
    sc, sp, var = pca_scores(Xs[gname][train_rows], Xe[gname])
    X_sm_clean = Despike().transform(sm.X[valid])
    X_ec_clean = Despike().transform(ec.X)
    sig = cls_signatures(X_sm_clean, sm.Y[valid].to_numpy())
    targets = list(sm.Y.columns)
    signatures = {a: sig[targets.index(a)] for a in ANALYTES}
    peak = {a: float(sm.wavenumbers[(sm.wavenumbers > 800) & (sm.wavenumbers < 1600)][
        np.argmax(signatures[a][(sm.wavenumbers > 800) & (sm.wavenumbers < 1600)])]) for a in ANALYTES}

    # ------------------------------------------------------------------
    # 4. Calibration design: correlated against independent analytes.
    # ------------------------------------------------------------------
    g, ac = sm.analyte("Glucose"), sm.analyte("Acetate")
    both = valid & ~np.isnan(g) & ~np.isnan(ac)
    btr, bte = both & (split == "train"), both & (split == "test")
    design_rows = []
    for analyte in ANALYTES:
        name, a = chosen[analyte]
        rows = design_experiment(Xs[name][btr], g[btr], ac[btr], sm.group[btr], Xs[name][bte], g[bte], ac[bte],
                                 chosen["Glucose"][1], chosen["Acetate"][1], n_draws=N_DRAWS)
        design_rows += [r for r in rows if r["analyte"] == analyte]   # each analyte with its own pipeline
    design = pd.DataFrame(design_rows)
    design.to_csv(res / "calibration_design.csv", index=False)
    design_summary = design.groupby(["analyte", "design"]).agg(
        training_correlation=("training_correlation", "mean"),
        rmsep_mean=("rmsep_test", "mean"), rmsep_sd=("rmsep_test", "std")).round(3).reset_index()
    print(design_summary.to_string(index=False), flush=True)
    from scipy.stats import wilcoxon
    design_test = {}
    for analyte in ANALYTES:
        part = design[design["analyte"] == analyte].pivot(index="draw", columns="design", values="rmsep_test")
        diff = part["correlated"] - part["independent"]
        design_test[analyte] = {"mean_increase": round(float(diff.mean()), 3),
                                "percent_increase": round(float(100 * diff.mean() / part["independent"].mean()), 1),
                                "draws_worse": int((diff > 0).sum()), "draws": int(len(diff)),
                                "wilcoxon_p": float(wilcoxon(part["correlated"], part["independent"]).pvalue)}

    # ------------------------------------------------------------------
    # 5. Model maintenance: adding E. coli samples.
    # ------------------------------------------------------------------
    upd = []
    for analyte in ANALYTES:
        name, a = chosen[analyte]
        y, ye = sm.analyte(analyte), ec.analyte(analyte)
        tr, me = masks[analyte]["train"], masks[analyte]["ecoli"]
        pool, held = me & (ec.source_split == "train"), me & (ec.source_split != "train")
        rows = update_experiment(Xs[name][tr], y[tr], Xe[name], np.nan_to_num(ye), ec.group, pool, held, a,
                                 n_draws=N_DRAWS)
        for r in rows:
            r.update(analyte=analyte, held_out_sd=float(ye[held].std()), held_out_spectra=int(held.sum()))
        upd += rows
        print(analyte, "update done", flush=True)
    update = pd.DataFrame(upd)
    update.to_csv(res / "model_update.csv", index=False)
    update_summary = update.groupby(["analyte", "policy", "samples_added"]).agg(
        rmsep_mean=("rmsep", "mean"), rmsep_sd=("rmsep", "std"), bias_mean=("bias", "mean"),
        flagged_mean=("fraction_flagged_99", "mean"), lv_mean=("latent_variables", "mean"),
        held_out_sd=("held_out_sd", "first")).reset_index().fillna(0).round(3)
    update_summary.to_csv(res / "model_update_summary.csv", index=False)

    # ------------------------------------------------------------------
    # 5b. Same protocol as the source study: E. coli cultivations split 8/8/8.
    # ------------------------------------------------------------------
    bench = {}
    for analyte in ANALYTES:
        name, a = chosen[analyte]
        y, ye = sm.analyte(analyte), ec.analyte(analyte)
        tr = masks[analyte]["train"]
        me = masks[analyte]["ecoli"]
        etr, eva, ete = (me & (ec.source_split == s) for s in ("train", "val", "test"))

        def scores(p, k):
            p = np.ravel(p)
            return {"rmse": round(rmse(ye[ete], p), 3), "mae": round(float(np.mean(np.abs(ye[ete] - p))), 3),
                    "latent_variables": int(k)}

        res_b = {"designed_only": scores(pls(a).fit(Xs[name][tr], y[tr]).predict(Xe[name][ete]), a)}
        _, k = select_pls(Xe[name][etr], ye[etr], Xe[name][eva], ye[eva], max_components=20)
        res_b["ecoli_train_only"] = scores(pls(k).fit(Xe[name][etr], ye[etr]).predict(Xe[name][ete]), k)
        Xt = np.vstack([Xs[name][tr]] + [Xe[name][etr]] * 10)
        yt = np.concatenate([y[tr]] + [ye[etr]] * 10)
        _, k = select_pls(Xt, yt, Xe[name][eva], ye[eva], max_components=25)
        res_b["designed_plus_ecoli_train_x10"] = scores(pls(k).fit(Xt, yt).predict(Xe[name][ete]), k)
        res_b["test_spectra"] = int(ete.sum())
        res_b["test_sd"] = round(float(ye[ete].std()), 3)
        bench[analyte] = res_b
    print(json.dumps(bench, indent=1), flush=True)

    summary.update(selected=final, calibration_design_test=design_test, source_protocol_comparison=bench, validation_vs_ecoli_rank_correlation=rank, published_split=leak,
                   pca_explained_variance=[round(float(v), 3) for v in var], signature_peak_800_1600=peak)
    (res / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ------------------------------------------------------------------
    # 6. Figures.
    # ------------------------------------------------------------------
    fig.spectra_and_signatures(sm.wavenumbers, X_sm_clean, X_ec_clean, signatures, figs)
    fig.selection_curves(curves, {k: v[0] for k, v in chosen.items()}, figs)
    fig.transfer(pred_fig, figs)
    fig.monitoring(mon_fig, figs)
    fig.pca(sc, sp, var, figs)
    fig.coefficients(coef_axes, coefs, figs)
    fig.update(update_summary, figs)
    fig.design(design, figs)


if __name__ == "__main__":
    main()
