# Paper Outline: Raman Bioprocess Chemometrics

Working plan for a research article built on the `raman-bioprocess-chemometrics` repository. All numbers come from its `results/` folder.

---

## Working title

**From designed mixtures to fermentation samples: calibration design, outlier monitoring, and model maintenance for Raman PLS models**

Alternative: *How many process samples does a Raman calibration need? A study of glucose and acetate in E. coli cultivations*

## Target journals (pick one; all indexed and peer reviewed)

| Journal | Fit |
|---|---|
| Journal of Raman Spectroscopy (Wiley) | Raman methods, chemometrics welcome |
| Analytical and Bioanalytical Chemistry (Springer) | Bioanalytical methods and validation |
| Biotechnology Progress (AIChE / Wiley) | PAT for bioprocesses |
| Chemometrics and Intelligent Laboratory Systems (Elsevier) | Methods focus: calibration design, monitoring, updating |

My first choice would be Journal of Raman Spectroscopy or Biotechnology Progress: the story is applied, and the data are bioprocess Raman.

## Contribution (what is new)

The source studies built the datasets and benchmarked models within each set. This paper connects them and asks deployment questions they did not report:

1. How a calibration on designed mixtures transfers to real fermentation samples on the same instrument, and why it fails.
2. Whether T² and Q monitoring identify the failed predictions.
3. A controlled measurement of what correlated calibration samples cost, using the independent design of the mixture data.
4. How many process samples an update needs, and how adding them changes the result (augment, weighted, process-only).
5. A like-for-like comparison of PLS with the published CNN on the published cultivation split.

## Structure

**Abstract** (about 250 words): problem, data, five findings with numbers, one-line implication.

**1. Introduction**
- Raman PAT for bioprocesses; calibration on prepared samples vs process samples
- Correlation between analytes in fermentations, and why designed calibrations exist
- Model lifecycle: monitoring and updating after deployment
- Gap: transfer, monitoring and updating not tested on a public pair of datasets
- Aims (the five questions)

**2. Materials and methods**
- 2.1 Data: both datasets, licence, instrument, data checks (saturated spectra, 1713 cm⁻¹ artefact, replicate overlap in the published split)
- 2.2 Preprocessing: spike removal, six pipelines
- 2.3 Models and selection: PLS, ridge, validation split by sample
- 2.4 Monitoring: T², Q, limits
- 2.5 Prediction intervals: split conformal
- 2.6 Calibration design experiment
- 2.7 Model maintenance experiment
- 2.8 Comparison with the source study
- 2.9 Software and reproducibility (repository, Zenodo DOI, tests)

**3. Results**
- 3.1 Spectra and analyte signatures (Fig. 1)
- 3.2 Model selection and performance on designed samples (Fig. 2, Table 1)
- 3.3 Transfer to E. coli samples (Fig. 3, Table 2)
- 3.4 Monitoring and PCA (Fig. 4)
- 3.5 Calibration design (Fig. 5)
- 3.6 Model maintenance (Fig. 6, Table 3)
- 3.7 PLS vs CNN (Table 4)

**4. Discussion**
- Why the transfer failed: matrix band near 879 cm⁻¹, background, PCA separation
- Monitoring as a gate before reporting a value
- Calibration design: weak analytes borrow from strong ones
- Updating: weighting matters; when to rebuild with process samples only
- PLS vs deep learning at this data size
- Limitations (from the README) and what would remove them

**5. Conclusions** (one paragraph)

**Data and code availability**: Hugging Face datasets (CC BY 4.0); repository and Zenodo DOI.

## Figures (main text)

1. Spectra and analyte signatures (`spectra.png`)
2. Validation curves (`selection.png`)
3. Predicted vs reference (`transfer.png`)
4. T² vs Q and PCA (`monitoring.png` + `pca_scores.png`)
5. Calibration design (`calibration_design.png`)
6. Model maintenance (`model_update.png`)

Supplementary: regression vectors (`coefficients.png`), full model selection table, per-draw results.

## Work to finish before submission

1. **Repeat the main results over several random splits** of the designed data (for example 5 seeds) and report mean ± SD, so no headline number rests on one split.
2. **Statistical tests:** paired Wilcoxon for the design experiment (already computed in `summary.json`); confidence intervals for the PLS vs CNN comparison by bootstrapping test samples.
3. **Write to the dataset authors** (TU Berlin, Lange and Cruz Bournazou) to confirm two details: whether the SubstrateMixRaman spectra came from the same spectrometer as the E. coli data, and whether cultivation identifiers and sampling times are available. Times would allow a time-ordered update experiment. It is also a courteous step before publishing on their data.
4. **Archive the repository on Zenodo** and cite the DOI in the paper.
5. **Literature review**: Raman glucose and acetate monitoring in E. coli and CHO, designed calibrations and spiking, model maintenance, calibration transfer. Verify every citation before it goes in.

## Authorship

Sole author (Oluwaseun O. Ajayi), unless someone contributes analysis or writing. Acknowledge the dataset authors and cite both of their papers.
