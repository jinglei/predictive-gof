## run_power_1d.R -- 1-D power for the NPLM Fig. 8 analogue (EPJC 82:275), headline configurations, 400 toys per run.
##   (a) nu* = 0, sigma_A = 0.10 and 0.15                       (sigma_A = 0.05 exists: seeds 1001 / 601)
##   (b) sigma_A = 0.05, one nuisance at +-sigma at a time:      (+s,0), (-s,0), (0,+s), (0,-s); nu_star = c(nu_n, nu_s)
## Signals NP1, NP2, NP3; one seed per (sigma_A, nu*) setting, shared across signals as in the existing runs.
## Also the nu* = 0 nulls still missing at sigma_A = 0.10 (dict, NN) and 0.15 (NN), so every power row has a null.
##   dict -- lambda_z = 1.3, damping "none"; seeds 1012-1017 (signal), 1018 (null 0.10)
##   nn   -- (1,4,1), clip 4 weights only, 30k epochs, taper kappa = 2, GPU duty 1 (no throttle); seeds 507-512 (signal),
##           513 / 514 (null 0.10 / 0.15)
## run (from the repository root):
##   nohup nice -n 10 Rscript run_power_1d.R dict > run_power_1d_dict.log 2>&1 &
##   nohup nice -n 10 Rscript run_power_1d.R nn   > run_power_1d_nn.log   2>&1 &
source("utils.R"); source("nn1d_test.R"); source("nn1d_trunc.R")
wit <- match.arg(commandArgs(trailingOnly = TRUE)[1], c("dict", "nn"))
B <- 400L; NC <- 8L
sets <- list(   # sigma_A, nu*, seed offset
  list(sA = 0.10, nu = c(0, 0),     k = 1L), list(sA = 0.15, nu = c(0, 0),     k = 2L),
  list(sA = 0.05, nu = c(+0.05, 0), k = 3L), list(sA = 0.05, nu = c(-0.05, 0), k = 4L),
  list(sA = 0.05, nu = c(0, +0.05), k = 5L), list(sA = 0.05, nu = c(0, -0.05), k = 6L))
nulls <- if (wit == "dict") list(list(sA = 0.10, seed = 1018L)) else
           list(list(sA = 0.10, seed = 513L), list(sA = 0.15, seed = 514L))
one <- function(signal, sg, sA, nu, seed) {
  cat(sprintf("\n==== %s: %s, sigma_A = %g, nu* = (%g, %g), seed %d ====\n", wit, if (signal) sg else "null",
              sA, nu[1], nu[2], seed))
  if (wit == "dict")
    run_tilt_experiment(mode = "pois", signal = signal, sig = sg, n_toys = B, nu_star = nu, sigma_A = c(sA, sA),
                        lambda_z = 1.3, seed = seed, cores = NC, chunk = 2L * NC, save_dir = "runs_pois",
                        bump_damping = "none")
  else
    run_nn1d_study(B = B, seed = seed, sig = sg, signal = signal, nu_star = nu, sigma_A = c(sA, sA),
                   clip = 4, bias_clip = FALSE, epochs = 30000, kappa = 2, duty = 1, batch = 100, cores = NC,
                   keep_outputs = FALSE)
}
for (n in nulls) one(FALSE, "NP1", n$sA, c(0, 0), n$seed)
for (s in sets) for (sg in c("NP1", "NP2", "NP3"))
  one(TRUE, sg, s$sA, s$nu, if (wit == "dict") 1011L + s$k else 506L + s$k)
