# cvgof: predictive goodness-of-fit tests for new-physics searches

This repository holds the code that reproduces the simulation results in

> J. Lei, *Neural-network-powered new-physics searches by predictive goodness of fit*.

The method fits a flexible learner, the *witness*, to the departure from a Reference model on
part of the data. The witness is a penalized dictionary or a neural network. The test evaluates
the witness's likelihood gain over the fitted Reference on the held-out part. The gain is
cross-fitted over K folds, centered at its expectation under the fitted null, and studentized.

The variance is computed analytically. It has separate terms for the estimated nuisance
parameters, for the finite Reference sample and for the randomness of the fits. Calibration
therefore needs no pseudo-experiments.

The benchmarks are those of NPLM (d'Agnolo et al., EPJC 81:89 (2021) and EPJC 82:275 (2022)):
- the one-dimensional exponential toys, with signals NP1, NP2 and NP3;
- the five-dimensional dilepton data set (Zenodo record 4442665).

## Layout

```
utils.R                 1-D pipeline: data, folds, null fits, dictionary witness, test statistic
nn1d_test.R / .py       1-D neural-network witness (R: export/assemble; Python: per-fold training)
nn1d_trunc.R            1-D null-tail taper of the network witness
run_*.R, launch_*.R     1-D drivers (one per group of runs; see below)
5d/utils_5d.R           5-D pipeline: HDF5 input, selection, scale nuisance, pools, data sets,
                        pushforward null fit, test statistic
5d/nn_test_5d.R         5-D neural-network witness (R: export/assemble, run_nn_study_5d)
5d/nn_witness_test.py   5-D per-fold network training on the GPU (batched over folds and data sets)
5d/train_delta_5d.py    nuisance log-ratio networks delta_1, delta_2 (NPLM's method)
5d/delta/delta_raw5_p8.json   the trained delta networks used for every 5-D result
5d/run_*.R              5-D drivers
5d/bench_nn_5d.py       timing benchmark of the 5-D network training
paper_epjc/scripts/     one script per table or figure of the paper
paper_epjc/tables/      the summary tables (CSV) behind the paper's tables
paper_epjc/figures/     the paper's figures
```

## Requirements

- R >= 4.4 with:
  - `glmnet`, `jsonlite`, `ggplot2` and `patchwork`;
  - an HDF5 reader: `rhdf5` (Bioconductor) or `hdf5r` (CRAN).

  `parallel` and `stats` are part of base R. Results were produced with R 4.4.2, glmnet 4.1.8,
  rhdf5 2.50.2, jsonlite 1.8.8, ggplot2 3.4.4 and patchwork 1.2.0.
- Python 3 with `numpy`, `pandas` and `torch`. Results were produced with torch 2.12.1 and
  numpy 2.3.5.
- A CUDA GPU for the network witnesses (an NVIDIA RTX 4090 was used). The dictionary witness
  runs on the CPU.

## Setup

1. Clone the repository. Run the 1-D scripts and the scripts in `paper_epjc/scripts/` from the
   repository root, and run the 5-D drivers from `5d/`. All paths are relative to these
   directories.
2. **5-D data.** Download the dilepton samples from Zenodo record 4442665
   (https://zenodo.org/records/4442665). Unpack them so that each sample is a folder of `.h5`
   files under `5d/data/`:
   - `5d/data/DiLepton_SM/DiLepton_SM*.h5`;
   - `5d/data/DiLepton_Zprime300/`;
   - `5d/data/DiLepton_EFT06_2/`.

   The first `load_sample_5d()` call caches each sample as RDS in `5d/cache/`. The file
   `DiLepton_SM59.h5` is empty at the source and is skipped.
3. **Nuisance networks.** The 5-D drivers load the trained networks in
   `5d/delta/delta_raw5_p8.json`, which are the ones behind the paper's numbers. To retrain them:
   - build the paired training sample with `make_delta_samples_5d(pools, paired = TRUE)`;
   - export it with `export_delta_samples_5d()` (`5d/utils_5d.R`, Sect. 1.7);
   - from `5d/`, run
     `python3 train_delta_5d.py --csv <sample>.csv.gz --out delta/<name> --epochs 2000`.

   A retrained network will differ slightly from the supplied one.

## Reproducing the paper

The runs are heavy: each setting has 400 data sets (160 for the 5-D signals), each data set has
10 folds, and the network witnesses train on the GPU. Launch them in the background, for example
`nohup nice -n 10 Rscript run_valid_1d.R dict > run_valid_1d_dict.log 2>&1 &`. Each study writes
a progress log in its output directory.

The drivers set the configurations, seeds and numbers of data sets used in the paper. Each
driver's header comment lists its runs and output paths.

### 1-D benchmark (from the repository root)

The Poisson model has N0 = 2000 and K = 10. The dictionary witness uses lambda_z = 1.3 with
damping "none". The network witness is (1,4,1), with weights clipped at 4, 30k epochs and a
null-tail taper with kappa = 2.

| Runs | Driver | Output |
|---|---|---|
| dictionary: null (seed 1002) and NP1–3 (seed 1001), sigma_A = 0.05 | `Rscript launch_pois_dmpnone.R` | `runs_pois/*_lam1p3_*_dmpnone.RData` |
| dictionary: lambda_z = 1.1 and 1.5, on the same data sets | `Rscript run_lambda_scan_1d.R` | `runs_pois/` |
| dictionary: nulls at ±sigma (sigma_A = 0.05, 0.15) and at 0 (sigma_A = 0.15) | `Rscript run_valid_1d_dict.R` | `runs_pois/` |
| network: null (seed 501) and NP1–3 (seed 601), for clip c = 3, 4 and 6 at 20k/30k/50k epochs | `Rscript run_nn1d_grid.R <c>` | `nn1d_test/` |
| network: nulls at ±sigma, sigma_A = 0.05 | `Rscript run_valid_1d.R nn` | `nn1d_test/` |
| network: nulls at ±sigma, sigma_A = 0.15 | `Rscript run_valid_1d_nn015.R` | `nn1d_test/` |
| power grid and nulls at sigma_A = 0.10, 0.15 | `Rscript run_power_1d.R dict`, `Rscript run_power_1d.R nn` | `runs_pois/`, `nn1d_test/` |

Use `run_valid_1d.R` with `nn` only; its `dict` branch is superseded by `run_valid_1d_dict.R`.

### 5-D benchmark (from `5d/`)

The network witness is NPLM's (5,5,5) network with all parameters clipped. The settings are
sigma_A = (0.025, 0.003) for (normalization, scale), N0 = 8700 and K = 10.

| Runs | Driver | Output |
|---|---|---|
| null (400 data sets, seed 201), Z′ with N_S = 120 (160, seed 61), EFT (160, seed 71), for clip c = 0.7, 1.0, 1.3 at 2000/3000/5000 epochs | `Rscript run_nn_sens_5d.R` | `5d/nn_test/sens_{null,zp,eft}_c*/` |
| nulls at ±sigma_s, ±sigma_n (c = 0.7, 3000 epochs; seeds 111–114) | `Rscript run_valid_5d.R plus` (and `minus`, `nplus`, `nminus`) | `5d/nn_test/sens_null_*_c0.7/` |

The headline configuration is c = 0.7 with 3000 epochs. The default arguments of
`run_nn_study_5d()` (`clip = 0.72`, `epochs = 2800`) belong to an earlier exploratory run; the
drivers override them.

The timing benchmark is
`python3 bench_nn_5d.py --toys 1 --clip 0.7 --epochs 500`, run from `5d/` with
`--toys 1` and `--toys 32`.

### Tables and figures (from the repository root)

Each script reads the run outputs above and writes into `paper_epjc/tables/` or
`paper_epjc/figures/`.

| Paper item | Script |
|---|---|
| 1-D null rates and median Z (Section 5) | `table_1d.R`, then `tex_tables_1d.R` |
| 1-D power grid (Section 5) | `table_1d_power.R`, `fig_1d_power.R` |
| 1-D null Z with and without the fitting-variance term | `fig_1d_null.R` |
| 5-D null rates and median Z (Section 6) | `table_5d.R`, then `tex_tables_5d.R` |
| 5-D comparison with NPLM | `table_5d_nplm.R`, `fig_5d_power.R` |
| 5-D null Z with and without the fitting-variance term | `fig_5d_null.R` |
| tuning grids (Appendix D) | `table_sensitivity.R`, `fig_tuning.R`, `tex_tables_sensitivity.R` |
| pushforward fits at the search bound (Appendix C) | `nufit_bounds_5d.R` |

Run each as `Rscript paper_epjc/scripts/<name>`. The shared helpers are `summ.R` (summaries and
Monte Carlo standard errors) and `figs.R` (plot style). The scripts overwrite the supplied
tables and figures, except `nufit_bounds_5d.R`, which refuses to overwrite its CSV.

### Notes on reproducibility

- Every data set is generated from its own seed, derived from the run seed and the data-set
  index, so the data sets do not depend on the number of cores.
- The dictionary witness is deterministic given the data.
- Network training on the GPU is not bit-for-bit deterministic. A rerun of a network study gives
  results that agree with the paper's up to Monte Carlo variation, not digit for digit.
- Some code comments refer to sections of the paper's theory notes (`Theory_Methods`) and to the
  experiment log (`Experiments.md`). These documents are not part of this repository; the paper
  contains the relevant definitions.

## Conventions

- `nu_star = c(nu_n, nu_s)` is the true nuisance value: the log-normalization and the
  log-scale (1-D: rate) shifts.
- `sigma_A` is the standard deviation of the auxiliary measurement.
- The reported statistic is `Z_fit` (network witness) or `Z_fit_Q` (1-D dictionary witness). It
  is approximately standard normal under the null, and large values indicate a departure.

## License

MIT; see `LICENSE`.
