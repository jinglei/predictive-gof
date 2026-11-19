## run_lambda_scan_1d.R -- 1-D dictionary lambda_z scan with bump_damping = "none", mirroring launch_pois_dmpnone.R
## (the lambda_z = 1.3 set) exactly: Poisson mode, N0 = 2000, sigma_A = (0.05, 0.05), K = 10, B = 400 (the default
## n_toys of that set, passed explicitly), cores 8, chunk 16, null seed 1002, NP1-3 seed 1001 -- so the scan is paired
## across lambda_z (same toys). Replaces the soft-damping lambda_z = 1.1 / 1.5 runs, which are not used in the paper.
## run (from the repository root):  nohup nice -n 10 Rscript run_lambda_scan_1d.R > run_lambda_scan_1d.log 2>&1 &
source("utils.R")
NC <- 8L; DIR <- "runs_pois"; SA <- c(0.05, 0.05); CH <- 2L * NC; B <- 400L
for (LAM in c(1.1, 1.5)) {
  run_tilt_experiment(mode = "pois", signal = FALSE, n_toys = B, sigma_A = SA, lambda_z = LAM, seed = 1002,
                      cores = NC, chunk = CH, save_dir = DIR, bump_damping = "none")
  ## -> pois_null_N2000_sA0p05_0p05_lam<L>_K10_s1002_dmpnone
  for (sg in c("NP1", "NP2", "NP3"))
    run_tilt_experiment(mode = "pois", signal = TRUE, sig = sg, n_toys = B, sigma_A = SA, lambda_z = LAM, seed = 1001,
                        cores = NC, chunk = CH, save_dir = DIR, bump_damping = "none")
  ## -> pois_alt_NP{1,2,3}_N2000_sA0p05_0p05_lam<L>_K10_s1001_dmpnone
}
