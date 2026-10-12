"""Loading the two public Raman data sets.

SubstrateMixRaman (Lange et al., 2026): 6,960 spectra of designed mixtures of
eight bioprocess substances, with each concentration drawn independently, so
that no two analytes are correlated by design.

E. coli fermentation (Lange et al., 2025): 379 spectra of supernatant samples
taken hourly during batch and fed-batch E. coli cultivations, with glucose and
acetate measured by a Cedex Bio HT analyzer.

Both share one wavenumber axis, which indicates the same spectrometer, and both
record two spectra per sample. Both are distributed on Hugging
Face under CC BY 4.0 as train.csv, val.csv and test.csv, with one column per
wavenumber followed by one column per analyte.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = {
    "substratemix": "chlange/SubstrateMixRaman",
    "ecoli": "chlange/RamanSpectraEcoliFermentation",
}
SPLITS = ("train", "val", "test")
SATURATION = 65535   # 16-bit detector maximum


@dataclass
class RamanSet:
    wavenumbers: np.ndarray   # [n_wavenumbers], cm^-1
    X: np.ndarray             # [n_spectra, n_wavenumbers], counts
    Y: pd.DataFrame           # [n_spectra, n_analytes], g/L (NaN where not measured)
    source_split: np.ndarray  # the split each row came from in the published files
    group: np.ndarray         # sample identifier: replicate spectra share a group

    def analyte(self, name):
        return self.Y[name].to_numpy(dtype=float)


def download(data_dir="data"):
    """Fetch both data sets from Hugging Face (needs internet and huggingface_hub)."""
    from huggingface_hub import hf_hub_download
    for folder, repo in SOURCES.items():
        target = Path(data_dir) / folder
        target.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            hf_hub_download(repo, f"{split}.csv", repo_type="dataset", local_dir=target)


def _is_wavenumber(column):
    try:
        float(column)
        return True
    except ValueError:
        return False


def sample_groups(Y):
    """Assign one group per distinct composition.

    Each sample was measured twice, and the two spectra carry identical
    reference values. Rows with identical values across all analytes are
    treated as one sample, so replicates are never split between training and
    test data.
    """
    keys = pd.DataFrame(np.round(Y.to_numpy(dtype=float), 6)).fillna(-1.0)
    return keys.groupby(list(keys.columns), sort=False).ngroup().to_numpy()


def load(name, data_dir="data"):
    folder = Path(data_dir) / name
    frames = [pd.read_csv(folder / f"{split}.csv") for split in SPLITS]
    source_split = np.concatenate([[s] * len(f) for s, f in zip(SPLITS, frames)])
    df = pd.concat(frames, ignore_index=True)
    wn_cols = [c for c in df.columns if _is_wavenumber(c)]
    y_cols = [c for c in df.columns if c not in wn_cols]
    Y = df[y_cols].astype(float).reset_index(drop=True)
    return RamanSet(
        wavenumbers=np.asarray(wn_cols, dtype=float),
        X=df[wn_cols].to_numpy(dtype=float),
        Y=Y,
        source_split=source_split,
        group=sample_groups(Y),
    )


def valid_spectra(X):
    """Finite, not flat, and not saturated at the detector maximum."""
    X = np.asarray(X, dtype=float)
    return np.isfinite(X).all(axis=1) & (X.std(axis=1) > 0) & (X.max(axis=1) < SATURATION)


def group_split(groups, fractions=(0.70, 0.15, 0.15), seed=0):
    """Split rows into train, validation and test sets by sample group."""
    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(groups))
    n_train = int(round(fractions[0] * len(unique)))
    n_val = int(round(fractions[1] * len(unique)))
    labels = np.empty(len(groups), dtype=object)
    for name, members in [("train", unique[:n_train]),
                          ("val", unique[n_train:n_train + n_val]),
                          ("test", unique[n_train + n_val:])]:
        labels[np.isin(groups, members)] = name
    return labels
