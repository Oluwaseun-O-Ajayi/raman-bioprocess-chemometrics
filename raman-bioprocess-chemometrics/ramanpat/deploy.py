"""Package a fitted model for use on new spectra, and apply it.

A deployed process model is more than regression coefficients. The bundle keeps
everything needed to reproduce a prediction and to judge whether it can be
trusted: the preprocessing pipeline, the PLS model, the monitoring limits, the
wavenumber axis the model expects, the calibration range, and a record of how
the model was built.
"""

import datetime as _dt

import joblib
import numpy as np

from ramanpat import __version__
from ramanpat.monitoring import q_residuals


def make_bundle(prep, model, monitor, wavenumbers, y_cal, rmse_val, pipeline_name, analyte, halfwidth95):
    return {
        "prep": prep,
        "model": model,
        "monitor": monitor,
        "wavenumbers": np.asarray(wavenumbers, dtype=float),
        "calibration_range": (float(np.min(y_cal)), float(np.max(y_cal))),
        "record": {
            "analyte": f"{analyte} (g/L)",
            "pipeline": pipeline_name,
            "latent_variables": int(model.n_components),
            "n_calibration": int(len(y_cal)),
            "rmse_validation": float(rmse_val),
            "interval95_halfwidth": float(halfwidth95),
            "built": _dt.date.today().isoformat(),
            "procchem_version": __version__,
        },
    }


def save(bundle, path):
    joblib.dump(bundle, path)


def load(path):
    return joblib.load(path)


def predict(bundle, X, wavenumbers, confidence=0.99):
    """Predictions with T2 and Q flags for each spectrum.

    Refuses spectra recorded on a different wavenumber axis, since silently
    applying a model to misaligned data is a common deployment failure.
    """
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    if wavenumbers.shape != bundle["wavenumbers"].shape or \
            not np.allclose(wavenumbers, bundle["wavenumbers"]):
        raise ValueError("wavenumber axis does not match the axis the model was built on")
    X = np.asarray(X, dtype=float)
    Xp = bundle["prep"].transform(X)
    model, monitor = bundle["model"], bundle["monitor"]
    t2 = monitor.t2(model.transform(Xp))
    q = q_residuals(model, Xp)
    pred = np.ravel(model.predict(Xp))
    lo, hi = bundle["calibration_range"]
    h = bundle["record"]["interval95_halfwidth"]
    return {
        "prediction": pred,
        "interval95_low": pred - h,
        "interval95_high": pred + h,
        "t2": t2,
        "q": q,
        "t2_flag": t2 > monitor.t2_limit(confidence),
        "q_flag": q > monitor.q_limit(confidence),
        "outside_calibration_range": (pred < lo) | (pred > hi),
    }
