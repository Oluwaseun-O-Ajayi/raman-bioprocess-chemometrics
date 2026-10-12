"""Apply a saved model to a CSV of Raman spectra.

The CSV must have one spectrum per row and the Raman shifts as column headers,
in the layout of the source data. Extra non-numeric columns are ignored.

    python scripts/predict.py spectra.csv --model models/glucose_pls.joblib --out predictions.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ramanpat import deploy  # noqa: E402
from ramanpat.data import _is_wavenumber  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spectra")
    parser.add_argument("--model", default=str(ROOT / "models" / "glucose_pls.joblib"))
    parser.add_argument("--out", default="predictions.csv")
    args = parser.parse_args()

    bundle = deploy.load(args.model)
    df = pd.read_csv(args.spectra)
    cols = [c for c in df.columns if _is_wavenumber(c)]
    result = pd.DataFrame(deploy.predict(bundle, df[cols].to_numpy(dtype=float), [float(c) for c in cols]))
    result["review"] = result["t2_flag"] | result["q_flag"] | result["outside_calibration_range"]
    result.to_csv(args.out, index=False)
    print(f"{bundle['record']['analyte']}: {len(result)} spectra, "
          f"{int(result['review'].sum())} flagged for review -> {args.out}")


if __name__ == "__main__":
    main()
