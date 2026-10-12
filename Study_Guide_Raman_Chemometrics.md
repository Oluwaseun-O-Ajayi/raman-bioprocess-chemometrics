# Chemometrics and Process Analytics: Study Guide

Private prep notes for process analytics interviews (J&J Advanced Process Analytics and similar roles). It pairs with the `raman-bioprocess-chemometrics` repository. Every number quoted here comes from that repository's `results/` folder, so you can defend each one.

---

## 1. Your 60-second answer to "What modelling have you done?"

> "I build calibration and prediction models from instrument data. At J&J I developed an automated pH method: a Python script reads the plate-reader signal from 3 µL samples in 48- and 96-tube racks and converts it to pH using a calibration built from standards. I benchmarked it against the pH meter, the gold standard, for accuracy and reproducibility, so teams could confirm pH during cell line development and purification without waiting for the end of the assay.
>
> Outside work I built two open-source modelling projects. One is a Raman chemometrics project: PLS models for glucose and acetate, calibrated on about 7,000 spectra of designed mixtures and tested on 24 E. coli cultivations, with T² and Q outlier monitoring and a study of how many process samples it takes to repair a calibration. The other benchmarks machine learning models, from Gaussian processes to graph neural networks, under noisy labels and distribution shift."

Then stop and let them pick which one to dig into.

**Before the interview, fill in from your own records** (don't guess): the pH range covered, how many standards, the agreement with the pH meter (mean difference, or R² and slope), and the reproducibility (SD or %CV of replicates). Those four numbers turn the pH story into a validated calibration model.

**Only say what is true.** On hands-on Raman: describe exactly what you did on the instrument in your committee member's lab, no more. Then: "The modelling side I've worked through end to end on public Raman bioprocess data in my repository."

---

## 2. The project in one breath

Two public Raman data sets from the same spectrometer (785 nm). Calibration: 6,960 spectra of designed mixtures of 8 substances with independently varied concentrations (glucose 0 to 20 g/L, acetate 0 to 2.1 g/L). Process test: 379 spectra of E. coli supernatant from 24 cultivations (glucose 0 to 4.1 g/L, acetate 0 to 1.6 g/L).

- **Selection:** six preprocessing pipelines × PLS and ridge, chosen on a validation set split by sample. Glucose: SNV on the full spectrum, 11 latent variables. Acetate: no scaling, 17 latent variables.
- **Designed test set:** glucose RMSEP 1.91 g/L (R² 0.91), acetate 0.23 g/L (R² 0.86).
- **E. coli:** glucose 1.42 g/L with a −1.1 g/L bias, acetate 0.31 g/L. Both no better than predicting the mean.
- **Monitoring:** T² flagged 55% of E. coli spectra for glucose; flagged spectra had RMSEP 1.76 g/L, unflagged 0.83. Every E. coli spectrum was flagged for acetate.
- **Calibration design:** correlated calibration samples (r = 0.91) raised acetate error by 30%.
- **Maintenance:** 10 E. coli samples, weighted ×10, cut glucose error from 1.41 to 0.53 g/L. A process-only model needed all 69 samples to reach 0.28 g/L.
- **Benchmark:** on the source study's split, PLS gave glucose MAE 0.22 g/L against their CNN's 0.27.

---

## 3. Core concepts

### 3.1 PCA (principal component analysis)

- **What it is:** a rotation of the data onto new axes (principal components) ordered by how much variance each explains. PC1 explains the most, PC2 the most of what is left, and so on. The components are uncorrelated.
- **Pieces:** X (spectra, centred) ≈ T Pᵀ + E.
  - **Scores (T):** where each sample sits on the new axes. One point per spectrum on a score plot.
  - **Loadings (P):** which wavenumbers make up each component. A loading looks like a spectrum.
  - **Residuals (E):** what the chosen components do not explain.
- **PCA is unsupervised.** It never sees y. It finds the biggest variation, which may have nothing to do with your analyte.
- **Uses:** exploring data, spotting clusters, outliers and drift, and building monitoring models (T² and Q work on PCA as well as PLS).
- **In the repo:** on the first two components (41% and 23% of variance), the E. coli spectra form their own cluster, apart from the designed mixtures. That one plot explains why the glucose model is biased and why monitoring flags so many E. coli spectra.

### 3.2 PLS (partial least squares regression)

- **The problem it solves:** spectra have hundreds or thousands of highly correlated wavenumbers (here 1,870). Neighbouring wavenumbers carry nearly the same information, so ordinary least squares is unstable and overfits.
- **The idea:** compress X into a few **latent variables (LVs)** chosen to have **maximum covariance with y**, then regress y on them. Unlike PCA, the components are steered by y.
- **Pieces:**
  - **Weights (W):** directions in X that covary most with y
  - **Scores (T):** samples projected onto those directions
  - **X-loadings (P)** and **y-loadings (q)**
  - **Regression vector (b):** the final coefficients per wavenumber, ŷ = Xb (after centring). Plot it and check that its peaks fall on the analyte's bands.
- **Mean-centring vs autoscaling:** for spectra, mean-centre only (`scale=False`). Autoscaling gives noisy, flat regions the same weight as real bands.
- **PLS1 vs PLS2:** PLS1 predicts one y (one model per analyte). PLS2 predicts several y together, which is useful when the analytes are correlated.
- **VIP scores** (variable importance in projection): a summary of which wavenumbers matter. VIP above about 1 is the usual rule of thumb for "important".

### 3.3 Choosing the number of latent variables

- Too few LVs: underfit, with high error everywhere.
- Too many: the model fits noise. Calibration error keeps falling, but cross-validation error stops improving or rises.
- **Rule used in the repo:** the smallest number of LVs whose validation RMSE is within 2% of the minimum. Other accepted rules: the first local minimum, or a statistical test such as Haaland and Thomas's F-test.
- **Sanity checks:** the regression vector should look chemical, not like noise, and each added LV should explain real variation.

### 3.4 Cross-validation

- **Purpose:** estimate how the model performs on samples it was not trained on, using only the calibration data.
- **Leave-one-out (LOO):** fit on n−1 samples, predict the one left out, repeat n times. Suits small sets. With thousands of spectra, as in the repo, a fixed validation set split by sample is enough.
- **k-fold, venetian blinds, contiguous blocks:** leave out groups. With time-ordered or replicate data, **leave out whole groups** (all replicates of a sample, a whole time block, or a whole batch). Otherwise near-duplicates sit in both training and test sets and the error looks better than it is.
- **No leakage:** preprocessing that learns from data (MSC reference spectrum, scaling, a fitted baseline) is fitted on calibration spectra only. Replicate spectra of one sample must stay on the same side of a split. The published split of the designed data broke this rule (458 of 659 test spectra shared a sample with training); the repo re-splits by sample.
- **CV is not validation.** It only tells you about samples like the calibration set. In the repo, glucose R² was 0.91 on designed test samples and below zero on the E. coli samples.

### 3.5 Batch-based validation

- **Why:** in bioprocessing, samples within one batch are correlated (same media lot, same cells, same probe position). Random CV across samples from the same batches is optimistic.
- **Leave-one-batch-out:** train on all batches but one, predict the held-out batch, repeat. This is the honest estimate of performance on the next batch.
- **Independent test batches:** keep whole runs out of model building and test once at the end. In the repo, the E. coli data are split by cultivation: process samples for updating come from 8 cultivations, and errors are measured on 16 other cultivations.
- **Time order matters too.** A deployed model only ever sees the past. The E. coli files have no timestamps, so the repo draws update samples at random within the training cultivations; time-ordered updates are listed as future work.

### 3.6 Spectral preprocessing

| Problem in the spectra | Fix | Note |
|---|---|---|
| Multiplicative scatter, path length or laser power changes | SNV, MSC, normalization to a band | SNV works per spectrum; MSC needs a reference spectrum fitted on calibration data |
| Constant baseline offset | First derivative | |
| Sloping baseline | Second derivative | Derivatives amplify noise, so combine with smoothing |
| Curved background (Raman fluorescence) | Polynomial or asymmetric least squares baseline (AsLS, airPLS), or derivatives | Fluorescence is the classic Raman headache |
| High-frequency noise | Savitzky-Golay smoothing | Window too wide blurs peaks; too narrow leaves noise |
| Cosmic ray spikes (Raman) | Spike detection and removal before anything else | One or two pixels wide; the repo's filter also removed a fixed artefact at 1713 cm⁻¹ |
| Uninformative or noisy regions | Region (variable) selection | In the repo the full spectrum won narrowly on validation for both analytes |

- **Savitzky-Golay parameters:** window length (odd number of points), polynomial order, derivative order.
- **Order matters.** A common sequence is spike removal, then region cut, then baseline or derivative, then scatter correction, then mean-centring.
- **Choose preprocessing by validation, not by how pretty the spectra look.** The repo scores every pipeline on the validation set.
- **A spike filter can damage real bands.** A first version with a 7-point window clipped the narrow sulfate band at 981 cm⁻¹; a 5-point window removed spikes and left the band alone. Always check what a filter changes.

### 3.7 Prediction errors and residuals

| Metric | What it measures |
|---|---|
| RMSEC | Error on the calibration samples themselves (always optimistic) |
| RMSECV | Error from cross-validation |
| RMSEP | Error on an independent test set; this is the number that matters |
| Bias | Mean of (predicted − reference); systematic offset |
| SEP | Standard deviation of prediction errors after bias is removed |
| R² / Q² | Fraction of variance explained in calibration / in CV |
| RPD | Standard deviation of reference values ÷ SEP; above about 3 is usually considered good for quantitation |

- **RMSEP² ≈ bias² + SEP².** If RMSEP is high, check whether it is mostly bias (fixable by a bias correction) or scatter (a deeper problem).
- **The reference method has error too.** You cannot beat the reference analyzer's own precision (the Cedex Bio HT in the E. coli data). Know its repeatability.
- **Residual plots:** plot residuals against reference value and against time. Patterns mean a missing effect. In the repo, glucose residuals on E. coli are mostly negative (bias −1.1 g/L), which points to a matrix difference, not random noise.
- **Compare RMSEP with the spread of the reference values.** An RMSEP larger than their standard deviation means no useful skill; R² below zero says the same.

### 3.8 Outliers: T², Q, and y-residuals

| Statistic | Asks | Example cause |
|---|---|---|
| Hotelling's T² | Is this sample extreme within the model plane? | Concentration beyond the calibrated range, an unusual mix of known components |
| Q (SPE, squared prediction error) | Does this spectrum contain something the model has never seen? | New compound, bubbles, fouled probe, fluorescence, instrument change |
| y-residual (calibration only) | Is the reference value inconsistent with the spectrum? | HPLC error, sample swap, timing mismatch |

- **Limits:** T² uses an F-distribution; Q uses Box's or Jackson and Mudholkar's approximation. 95% for warnings and 99% for action is a common convention.
- **Influence plot:** T² (or leverage) against Q. Points high on both axes are the most worrying.
- **In calibration:** investigate an outlier before removing it. Remove it only with a documented reason, such as a known sample prep error.
- **In deployment:** a flagged prediction is reported as "not reliable", not quietly used.
- **In the repo:** the glucose model flagged 3.5% of designed test spectra and 55% of E. coli spectra, mostly on T². Flagged E. coli spectra had RMSEP 1.76 g/L, unflagged 0.83 g/L, so the flags pointed at the bad predictions. The acetate model flagged every E. coli spectrum. The point to make: **"The monitoring told me which predictions to distrust before I compared them with the reference."**

### 3.9 Extrapolation

- PLS is a linear interpolator. Outside the calibrated range (or outside the spectral space), predictions are not trustworthy.
- Extrapolation also happens in spectral space: the E. coli glucose values sit inside the calibrated range, but the matrix does not, and the model failed.
- **Design fix:** the calibration must span the full process range plus a margin, and include the process variability (other components, media, cell debris, temperature).

---

## 4. Raman chemometrics specifics

- **What Raman measures:** inelastic scattering from molecular vibrations. It's complementary to infrared.
- **Water is a weak Raman scatterer.** That is the big reason Raman suits aqueous bioprocesses: it can measure glucose, lactate, glutamine, ammonium and others in situ through an immersion probe.
- **Bands in the repo:** the glucose signature estimated from the designed mixtures peaks at 1124 cm⁻¹, and acetate at 928 cm⁻¹ with a second band near 1415 cm⁻¹, matching the literature. Sulfate gives the sharp band at 981 cm⁻¹. The E. coli supernatant has an extra band near 879 cm⁻¹ that the mixtures lack.
- **Raman-specific problems:**
  - Fluorescence background, which is large and curved: baseline correction or derivatives
  - Cosmic ray spikes: remove them first
  - Laser power drift and probe changes: normalization
  - Low signal: longer integration and averaging scans, a trade-off against time resolution
- **Correlated analytes, the trap you measured:** in a fermentation, substrates fall while products rise. A model can predict one analyte by tracking another. In the repo, calibrations with glucose and acetate correlated at r = 0.91 raised acetate error on independent samples by 30%; glucose, the stronger signal, rose only 4%. The weak analyte borrows from the strong one.
  - Fixes: designed or spiked calibration samples that break the correlation, runs with different feeding profiles, and a regression vector that peaks on the analyte's own bands.

### Building a PLS model from process data

1. **Define the purpose:** the analyte, the required accuracy, and how the result will be used (monitoring only, or feedback control such as automated glucose feeding).
2. **Collect calibration data that span the space:** several runs, different scales or operating conditions, spiked samples to break correlations, and the full concentration range.
3. **Align spectra with reference samples:** match timestamps, account for the delay between sampling and quench, and average spectra over a window around each sample.
4. **Preprocess, then choose preprocessing and LVs by cross-validation**, leaving out whole batches.
5. **Validate on independent batches** never used for building. Report RMSEP, bias, and performance across the range.
6. **Set monitoring limits** (T², Q) from the calibration data.
7. **Document:** data used, outliers removed and why, the final model, and its validation.

---

## 5. Deployment and lifecycle management

- **A deployed model is a package, not a set of coefficients:** the preprocessing, model, monitoring limits, expected wavenumber axis, calibration range and a build record. `ramanpat/deploy.py` in the repo does exactly this, and refuses spectra on the wrong wavenumber axis.
- **Every prediction carries a status:** the value plus T², Q and a range check. Flagged predictions go for review.
- **Ongoing performance verification:** compare predictions with periodic reference samples. Track the bias and the error on a control chart.
- **Triggers for review or recalibration:**
  - Repeated T² or Q alarms
  - Drift in bias against reference samples
  - New raw material suppliers or media lots
  - Process changes (scale, cell line, feeding strategy)
  - Instrument or probe service or replacement
- **Update options, from light to heavy:**
  1. Bias or slope correction
  2. Add new samples and refit
  3. Calibration transfer between instruments (piecewise direct standardization, EPO)
  4. Full rebuild
- **Validate every update before it goes live**, on data it did not use. In the repo, adding 10 process samples without weighting barely helped (1.41 to 1.30 g/L); weighting them ×10 cut the error to 0.53 g/L. How you add samples matters as much as how many.
- **When to rebuild instead:** with all 69 process samples, a model on process samples alone (0.28 g/L) beat every update of the designed calibration.
- **Change control and versioning:** each model version is documented, approved, and traceable to its data.
- **Regulatory frame (know the names, not every clause):**
  - FDA PAT guidance (2004): the framework for using process analyzers in pharma manufacturing
  - ICH Q8(R2), Q9, Q10: design space, quality risk management, the pharmaceutical quality system
  - ICH Q2(R2) and Q14 (2023): validation and development of analytical procedures, including multivariate methods
  - The ICH Q8/Q9/Q10 Points to Consider classify models by impact (low, medium, high); a model used for control or release needs more validation than one used for monitoring
  - 21 CFR Part 11: electronic records and data integrity
- **Software you may hear about:** SIMCA, PLS_Toolbox and Solo, The Unscrambler for modelling; SynTQ, SIPAT and similar PAT data managers for deployment. Your answer: "I work in Python with scikit-learn and chemotools, and I'm comfortable learning your platform."

---

## 6. Likely questions and strong answers

**Why PLS and not ordinary regression?** 1,870 correlated wavenumbers. Ordinary least squares is unstable. PLS compresses to a few latent variables that covary with y.

**PCA vs PLS?** PCA finds maximum variance in X and ignores y. PLS finds directions in X with maximum covariance with y. PCA explores and monitors; PLS predicts.

**How did you choose the number of latent variables?** Validation RMSE for 1 to 25 LVs, on samples split so replicates never crossed sets. I took the smallest number within 2% of the minimum, then checked that the glucose regression vector peaked at the 1125 cm⁻¹ band.

**Why not a neural network?** I tested PLS against the source study's CNN on their own split. PLS trained on 8 cultivations gave glucose MAE 0.22 g/L against the CNN's 0.27; for acetate the CNN was better (0.06 against 0.084). PLS is simpler, interpretable, and easier to validate, so I would start there and move to a neural network only if it clearly wins on held-out batches.

**Your model was accurate on designed samples and failed on the fermentation. Why?** The fermentation supernatant has a different matrix: a band near 879 cm⁻¹ and a higher background that the designed mixtures never had. PCA shows the E. coli spectra in their own cluster, and T² flagged 55% of them. The glucose predictions followed the trend but sat 1.1 g/L low.

**How did you fix it?** I added process samples. Ten samples, weighted ×10 so they count among 4,872 designed spectra, cut the error from 1.41 to 0.53 g/L on 16 cultivations the update never saw. Without weighting, the same 10 samples did almost nothing.

**What does a high Q mean? A high T²?** High Q: the spectrum has a feature the model can't reconstruct, something new. High T²: within the model's space, but extreme, such as an unusual combination of known components.

**Did the monitoring work?** For glucose, flagged spectra had RMSEP 1.76 g/L and unflagged 0.83, so the flags pointed at the bad predictions. For acetate, every E. coli spectrum was flagged, which correctly says that model should not be used on this process.

**Why did acetate fail?** Its signal is weak: E. coli acetate is mostly below 0.4 g/L, and the designed-data error is 0.23 g/L. The regression vector also leans on broad background, which differs between matrices. Calibrations where acetate tracks glucose make it worse (+30%).

**What preprocessing would you use for Raman?** Spike removal, region selection, baseline correction or a derivative with Savitzky-Golay smoothing, then SNV or normalization, then mean-centring. The final choice is made on validation data, not by eye, and I check what each filter changes.

**How do you know the model is predicting glucose and not something correlated with it?** Check the regression vector against the glucose spectrum, use calibration samples that break the correlation, and test on runs with different profiles. My design experiment measured what happens when you don't.

**How would you validate for a GMP setting?** Independent batches, accuracy across the range, precision (repeatability), specificity, robustness, a defined range, and documented limits. Then ongoing verification after deployment under change control.

**What would make you recalibrate?** Persistent T² or Q alarms, bias drift against reference samples, raw material or process changes, or instrument service.

**What is RMSEP vs bias?** RMSEP is total error. Bias is the average offset. On E. coli, glucose RMSEP was 1.42 g/L with a bias of −1.12, so most of the error was offset, which is why a few process samples fixed it.

---

## 7. A 20-minute presentation outline

| Minutes | Slide | Message |
|---|---|---|
| 0-2 | Who I am | Analytical chemist; I build calibration models from instrument data and make them reliable |
| 2-6 | J&J pH method | The problem (waiting on end-of-assay pH), the calibration against standards, the benchmark against the pH meter, the impact on cell line development and purification. Use your real numbers |
| 6-8 | The process analytics problem | Calibrate on designed mixtures, use on fermentation samples. Show the spectra and the two data sets |
| 8-10 | Model building | Preprocessing, validation curves, glucose regression vector on the 1125 cm⁻¹ band |
| 10-12 | Transfer | Predicted vs reference: R² 0.91 on designed samples, below zero on E. coli |
| 12-14 | Monitoring | T²/Q plot and PCA. Flags pointed at the bad predictions |
| 14-16 | Calibration design | Correlated samples cost acetate 30% |
| 16-18 | Lifecycle | 10 weighted process samples fixed glucose; deployment bundle with flags; PLS vs CNN |
| 18-20 | What I'd do with your data | Batch-wise validation, spiked calibration runs, monitoring and change control. Then questions |

Practise it out loud twice with a timer. Each slide gets one figure and one sentence of takeaway.

---

## 8. Numbers to memorize

- Designed data: 6,960 spectra, 3,433 samples, 8 analytes. E. coli: 379 spectra, 190 samples, 24 cultivations.
- Glucose model: full spectrum, SNV, 11 LVs. Designed test RMSEP 1.91 g/L, R² 0.91. E. coli RMSEP 1.42, bias −1.12.
- Acetate model: full spectrum, no scaling, 17 LVs. Designed test 0.23 g/L, R² 0.86. E. coli 0.31.
- Monitoring (glucose): 3.5% of designed test flagged, 55% of E. coli; RMSEP flagged 1.76 vs unflagged 0.83.
- Design: acetate 0.254 → 0.329 g/L (+30%) with correlated calibration; glucose +4%.
- Update (glucose): 1.41 → 0.53 g/L with 10 weighted samples; process-only model 0.28 with all 69.
- Benchmark: glucose MAE 0.22 (PLS) vs 0.27 (CNN); acetate 0.084 vs 0.06.
- Bands: glucose 1124 cm⁻¹, acetate 928 and 1415 cm⁻¹, sulfate 981 cm⁻¹.
