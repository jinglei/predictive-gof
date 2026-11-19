## run_valid_1d_nn015.R -- 1-D network null validity at sigma_A = 0.15, the four one-at-a-time shifted points
## (completes Table tab:1d-null; the nu* = 0 point at 0.15 exists: seed 514, run_power_1d.R).
## nu_star = c(nu_n, nu_s): (+0.15, 0), (-0.15, 0), (0, +0.15), (0, -0.15); seeds 515-518.
## Headline configuration, as run_valid_1d.R nn: (1,4,1), clip 4 weights only, 30k epochs, taper kappa = 2, B = 400,
## sigma_A = (0.15, 0.15), N0 = 2000, K = 10; GPU duty 1 (as run_power_1d.R).
## run (from the repository root):  nohup nice -n 10 Rscript run_valid_1d_nn015.R > run_valid_1d_nn015.log 2>&1 &
source("utils.R"); source("nn1d_test.R"); source("nn1d_trunc.R")
SA <- c(0.15, 0.15); NC <- 8L
pts <- list(c(+0.15, 0), c(-0.15, 0), c(0, +0.15), c(0, -0.15))
for (i in seq_along(pts)) {
  nu <- pts[[i]]
  cat(sprintf("\n==== nn: null, sigma_A = 0.15, nu* = (%g, %g), seed %d ====\n", nu[1], nu[2], 514L + i))
  run_nn1d_study(B = 400, seed = 514L + i, signal = FALSE, nu_star = nu, sigma_A = SA,
                 clip = 4, bias_clip = FALSE, epochs = 30000, kappa = 2, duty = 1, batch = 100, cores = NC,
                 keep_outputs = FALSE)
}
