"""Figures for the README."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, GREEN, GREY, RED, PURPLE = "#1b6ca8", "#d98c1f", "#2a9d6f", "#8a8a8a", "#c44e52", "#7b5ea7"
BANDS = {"Glucose": [1125], "Acetate": [928, 1415]}
ANALYTES = ("Glucose", "Acetate")


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def spectra_and_signatures(wn, X_cal, X_proc, signatures, out):
    """Median spectra of both sets, and the analyte signatures from the designed data."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    axes[0].plot(wn, np.median(X_cal, axis=0), color=BLUE, lw=1, label="designed mixtures (median)")
    axes[0].plot(wn, np.median(X_proc, axis=0), color=RED, lw=1, label="E. coli supernatant (median)")
    axes[0].set_ylabel("counts")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_title("Raman spectra, 785 nm")
    for name, color in [("Glucose", GREEN), ("Acetate", PURPLE)]:
        axes[1].plot(wn, signatures[name], color=color, lw=1, label=f"{name}, counts per g/L")
        for b in BANDS[name]:
            axes[1].axvline(b, color=color, ls=":", lw=0.8)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xlim(400, 1800)
    axes[1].set_ylim(-60, 160)
    axes[1].set_xlabel("Raman shift (cm$^{-1}$)")
    axes[1].set_ylabel("signal per g/L")
    axes[1].set_title("Analyte signatures from the designed mixtures (dotted: literature bands; y-axis clipped)")
    axes[1].legend(fontsize=8, frameon=False)
    _save(fig, out / "spectra.png")


def selection_curves(curves, chosen, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, analyte in zip(axes, ANALYTES):
        part = curves[curves["analyte"] == analyte]
        for name, group in part.groupby("pipeline", sort=False):
            ax.plot(group["latent_variables"], group["rmse_val"], marker="o", ms=2.5,
                    lw=2.4 if name == chosen[analyte] else 1.0, label=name)
        ax.set_title(analyte)
        ax.set_xlabel("PLS latent variables")
        ax.set_ylabel("validation RMSE (g/L)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False)
    _save(fig, out / "selection.png")


def transfer(pred, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, analyte in zip(axes, ANALYTES):
        p = pred[analyte]
        ax.scatter(p["cal_ref"], p["cal_pred"], s=5, color=GREY, alpha=0.4, label="designed mixtures, test set")
        ax.scatter(p["proc_ref"], p["proc_pred"], s=12, color=RED, alpha=0.8, label="E. coli supernatant")
        lim = [min(p["cal_ref"].min(), p["proc_pred"].min()) - 0.5, p["cal_ref"].max() * 1.05]
        ax.plot(lim, lim, color="black", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(analyte)
        ax.set_xlabel("reference (g/L)")
        ax.set_ylabel("predicted (g/L)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    _save(fig, out / "transfer.png")


def monitoring(mon, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, analyte in zip(axes, ANALYTES):
        m = mon[analyte]
        ax.scatter(m["cal_t2"], m["cal_q"], s=5, color=GREY, alpha=0.4, label="designed mixtures, test set")
        ax.scatter(m["proc_t2"], m["proc_q"], s=12, color=RED, alpha=0.8, label="E. coli supernatant")
        ax.axvline(m["t2_limit"], color=ORANGE, ls="--", lw=1, label="99% limits")
        ax.axhline(m["q_limit"], color=ORANGE, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Hotelling's T$^2$")
        ax.set_ylabel("Q residual")
        ax.set_title(f"{analyte} model")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    _save(fig, out / "monitoring.png")


def pca(scores_cal, scores_proc, var, out):
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(scores_cal[:, 0], scores_cal[:, 1], s=4, color=GREY, alpha=0.35, label="designed mixtures")
    ax.scatter(scores_proc[:, 0], scores_proc[:, 1], s=12, color=RED, alpha=0.8, label="E. coli supernatant")
    ax.set_xlabel(f"PC1 ({var[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.0%} of variance)")
    ax.set_title("PCA after preprocessing (glucose pipeline)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    _save(fig, out / "pca_scores.png")


def coefficients(axes_by_analyte, coefs, out):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6))
    for ax, analyte, color in zip(axes, ANALYTES, (GREEN, PURPLE)):
        ax.plot(axes_by_analyte[analyte], coefs[analyte], color=color, lw=0.8)
        for b in BANDS[analyte]:
            ax.axvline(b, color="black", ls=":", lw=0.8)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlim(400, 1800)
        ax.set_ylabel("coefficient")
        ax.set_title(f"{analyte} PLS regression vector (dotted: literature bands)")
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("Raman shift (cm$^{-1}$)")
    _save(fig, out / "coefficients.png")


def update(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {"augment": (BLUE, "add samples to calibration"),
              "augment_weighted": (GREEN, "add samples, weighted x10"),
              "local": (ORANGE, "process samples only")}
    for ax, analyte in zip(axes, ANALYTES):
        part = summary[summary["analyte"] == analyte]
        base = part.loc[part["policy"] == "none", "rmsep_mean"].iloc[0]
        ax.axhline(base, color=GREY, ls="--", lw=1.2, label="designed calibration only")
        for policy, (color, label) in styles.items():
            p = part[part["policy"] == policy].sort_values("samples_added")
            ax.plot(p["samples_added"], p["rmsep_mean"], marker="o", ms=3, color=color, label=label)
            ax.fill_between(p["samples_added"], p["rmsep_mean"] - p["rmsep_sd"],
                            p["rmsep_mean"] + p["rmsep_sd"], color=color, alpha=0.15)
        ax.axhline(part["held_out_sd"].iloc[0], color="black", ls=":", lw=1, label="SD of held-out reference")
        ax.set_title(analyte)
        ax.set_xlabel("E. coli samples used")
        ax.set_ylabel("RMSEP on held-out E. coli samples (g/L)")
        ax.set_ylim(0, None)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False)
    _save(fig, out / "model_update.png")


def design(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, analyte in zip(axes, ANALYTES):
        part = df[df["analyte"] == analyte]
        data = [part.loc[part["design"] == d, "rmsep_test"] for d in ("independent", "correlated")]
        ax.boxplot(data, widths=0.5)
        ax.set_xticks([1, 2], ["independent", "correlated (r = 0.9)"])
        ax.set_ylabel("RMSEP on independent test set (g/L)")
        ax.set_title(analyte)
        ax.grid(alpha=0.3, axis="y")
    _save(fig, out / "calibration_design.png")
