## run_valid_1d_dict.R -- 1-D dictionary null validity (lambda_z = 1.3, damping "none", N0 = 2000, K = 10).
## B = 400 = n_toys of the nu* = 0 headline null runs_pois/pois_null_N2000_sA0p05_0p05_lam1p3_K10_s1002_dmpnone.RData.
## NPLM points (EPJC 82:275, Fig. 5 / Fig. 8): one nuisance shifted by +-1 sigma at a time, nu_star = c(nu_n, nu_s).
##   sigma_A = 0.05: (+-0.05, 0), (0, +-0.05)            seeds 1003-1006   (nu* = 0 is the existing s1002 run)
##   sigma_A = 0.15: (0, 0), (+-0.15, 0), (0, +-0.15)    seeds 1007-1011
## Supersedes the "dict" branch of run_valid_1d.R (that file is left as launched for the NN set).
## run (from the repository root):  nohup nice -n 10 Rscript run_valid_1d_dict.R > run_valid_1d_dict.log 2>&1 &
source("utils.R")
B <- 400L; NC <- 8L
runs <- list(
  list(sA = 0.05, nu = c(+0.05, 0),  seed = 1003L), list(sA = 0.05, nu = c(-0.05, 0),  seed = 1004L),
  list(sA = 0.05, nu = c(0, +0.05),  seed = 1005L), list(sA = 0.05, nu = c(0, -0.05),  seed = 1006L),
  list(sA = 0.15, nu = c(0, 0),      seed = 1007L),
  list(sA = 0.15, nu = c(+0.15, 0),  seed = 1008L), list(sA = 0.15, nu = c(-0.15, 0),  seed = 1009L),
  list(sA = 0.15, nu = c(0, +0.15),  seed = 1010L), list(sA = 0.15, nu = c(0, -0.15),  seed = 1011L))
for (r in runs) {
  cat(sprintf("\n==== dict: sigma_A = %g, nu* = (%g, %g), seed %d, B = %d ====\n", r$sA, r$nu[1], r$nu[2], r$seed, B))
  run_tilt_experiment(mode = "pois", signal = FALSE, n_toys = B, nu_star = r$nu, sigma_A = c(r$sA, r$sA),
                      lambda_z = 1.3, seed = r$seed, cores = NC, chunk = 2L * NC, save_dir = "runs_pois",
                      bump_damping = "none")
}
