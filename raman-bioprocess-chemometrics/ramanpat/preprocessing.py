"""Preprocessing pipelines for Raman spectra.

Every pipeline starts with cosmic-ray removal. All steps except MSC act on each
spectrum separately; MSC learns a mean reference spectrum from the calibration
data. Each pipeline is fitted on calibration spectra only and then applied
unchanged to new spectra.
"""

from chemotools.derivative import SavitzkyGolay
from chemotools.feature_selection import RangeCut
from chemotools.scatter import MultiplicativeScatterCorrection, StandardNormalVariate
from chemotools.smooth import SavitzkyGolayFilter
from sklearn.pipeline import Pipeline

from ramanpat.raman import Despike

# Fingerprint region: covers the glucose band near 1125 cm^-1 and the acetate
# bands near 928 and 1415 cm^-1, and stops below the water bending band near
# 1640 cm^-1 and a fixed instrument artefact near 1712 cm^-1.
REGION = (800.0, 1600.0)


def make_pipelines(wavenumbers, region=REGION):
    """The six pipelines compared, defined before looking at any results.

    - full_raw: whole spectrum, spikes removed, nothing else
    - full_snv: whole spectrum, standard normal variate
    - region_snv: fingerprint region, SNV
    - region_smooth_snv: fingerprint region, Savitzky-Golay smoothing (11 points), SNV
    - region_sg1_snv: fingerprint region, Savitzky-Golay first derivative (15 points), SNV
    - region_msc: fingerprint region, multiplicative scatter correction
    """
    start, end = region

    def cut():
        return ("region", RangeCut(start=start, end=end, x_axis=wavenumbers))

    return {
        "full_raw": Pipeline([("despike", Despike())]),
        "full_snv": Pipeline([("despike", Despike()), ("snv", StandardNormalVariate())]),
        "region_snv": Pipeline([("despike", Despike()), cut(), ("snv", StandardNormalVariate())]),
        "region_smooth_snv": Pipeline([
            ("despike", Despike()), cut(),
            ("smooth", SavitzkyGolayFilter(window_length=11, polyorder=2)),
            ("snv", StandardNormalVariate()),
        ]),
        "region_sg1_snv": Pipeline([
            ("despike", Despike()), cut(),
            ("sg1", SavitzkyGolay(window_length=15, polyorder=2, deriv=1)),
            ("snv", StandardNormalVariate()),
        ]),
        "region_msc": Pipeline([("despike", Despike()), cut(), ("msc", MultiplicativeScatterCorrection())]),
    }


def output_axis(fitted_pipeline, wavenumbers):
    """Wavenumbers that survive a fitted pipeline."""
    steps = dict(fitted_pipeline.steps)
    if "region" in steps:
        cut = steps["region"]
        return wavenumbers[cut.start_index_:cut.end_index_]
    return wavenumbers
