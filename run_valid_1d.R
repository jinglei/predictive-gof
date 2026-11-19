## run_valid_1d.R -- 1-D null validity at the four off-centre nuisance points of NPLM (EPJC 82:275, Fig. 5 and Fig. 8):
## one nuisance shifted by +-1 sigma at a time, (nu_n*, nu_s*) = (+s, 0), (-s, 0), (0, +s), (0, -s), s = sigma_A = 0.05
## (nu_star = c(nu_n, nu_s): nu_n scales the expected count, nu_s the rate; utils.R gen_data_pois).
## Headline configurations, 400 toys per point, N0 = 2000, K = 10:
##   dict -- lambda_z = 1.3, damping "none"; seeds 1003-1006   -> runs_pois/pois_null_..._nu<n>_<s>_..._dmpnone.RData
##   nn   -- (1,4,1), clip 4 weights only, 30k epochs, taper kappa = 2, GPU duty 0.7; seeds 503-506
##           -> nn1d_test/null_nu<n>_<s>_c4w_e30000_tap2_s<seed>/results.rds
## run (from the repository root; dict is CPU only, nn uses the GPU):
##   nohup nice -n 10 Rscript run_valid_1d.R dict > run_valid_1d_dict.log 2>&1 &
##   nohup nice -n 10 Rscript run_valid_1d.R nn   > run_valid_1d_nn.log   2>&1 &
source("utils.R"); source("nn1d_test.R"); source("nn1d_trunc.R")
wit <- match.arg(commandArgs(trailingOnly = TRUE)[1], c("dict", "nn"))
SA <- c(0.05, 0.05); NC <- 8L
pts <- list(n_plus = c(+SA[1], 0), n_minus = c(-SA[1], 0), s_plus = c(0, +SA[2]), s_minus = c(0, -SA[2]))
for (i in seq_along(pts)) {
  nu <- pts[[i]]
  cat(sprintf("\n==== %s: %s, nu* = (%g, %g) ====\n", wit, names(pts)[i], nu[1], nu[2]))
  if (wit == "dict")
    run_tilt_experiment(mode = "pois", signal = FALSE, nu_star = nu, sigma_A = SA, lambda_z = 1.3, seed = 1002L + i,
                        cores = NC, chunk = 2L * NC, save_dir = "runs_pois", bump_damping = "none")
  else
    run_nn1d_study(B = 400, seed = 502L + i, signal = FALSE, nu_star = nu, sigma_A = SA,
                   clip = 4, bias_clip = FALSE, epochs = 30000, kappa = 2, duty = 0.7, batch = 100, cores = NC,
                   keep_outputs = FALSE)
}
