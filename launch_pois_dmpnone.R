## launch_pois_dmpnone.R -- 1-D Poisson-mode null + NP1/NP2/NP3 with bump_damping = "none",
## same settings and seeds as the soft-damping runs in Experiments.md (lambda = 1.3 block):
##   N0 = 2000, sigma_A = (0.05, 0.05), lambda_z = 1.3, K = 10, 400 toys, null seed 1002, alt seed 1001
## run:  nohup nice -n 10 Rscript launch_pois_dmpnone.R > launch_pois_dmpnone.log 2>&1 &
source("utils.R")
NC  <- 8L; DIR <- "runs_pois"; dir.create(DIR, showWarnings = FALSE)
LAM <- 1.3; SA <- c(0.05, 0.05); CH <- 2L * NC
r0 <- run_tilt_experiment(mode = "pois", signal = FALSE, sigma_A = SA, lambda_z = LAM, seed = 1002,
                          cores = NC, chunk = CH, save_dir = DIR, bump_damping = "none")
## -> pois_null_N2000_sA0p05_0p05_lam1p3_K10_s1002_dmpnone
for (sg in c("NP1", "NP2", "NP3"))
  run_tilt_experiment(mode = "pois", signal = TRUE, sig = sg, sigma_A = SA, lambda_z = LAM, seed = 1001,
                      cores = NC, chunk = CH, save_dir = DIR, bump_damping = "none")
## -> pois_alt_NP{1,2,3}_N2000_sA0p05_0p05_lam1p3_K10_s1001_dmpnone
