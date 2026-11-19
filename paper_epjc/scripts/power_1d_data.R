## power_1d_data.R -- the 1-D power grid (NP1-3 x NN / dictionary x nuisance setting), read from saved outputs.
## Settings: nu* = 0 at sigma_A = 0.05 / 0.10 / 0.15, and the four one-at-a-time +-sigma points at sigma_A = 0.05.
## Producing runs: sigma_A = 0.05, nu* = 0 -- run_nn1d_grid.R 4 (seed 601), launch_pois_dmpnone.R (seed 1001);
## all other settings -- run_power_1d.R (NN seeds 507-512, dictionary seeds 1012-1017).
## Defines power_1d(): one row per (witness, signal, setting) with summ_z() columns.
source("paper_epjc/scripts/summ.R")
POWER_SETS <- data.frame(
  setting = c("0", "0", "0", "+sigma_n", "-sigma_n", "+sigma_s", "-sigma_s"),
  sigma_A = c(0.05, 0.10, 0.15, 0.05, 0.05, 0.05, 0.05),
  nn_tag  = c(NA, "nu0", "nu0", "nu0.05_0", "nu-0.05_0", "nu0_0.05", "nu0_-0.05"),
  nn_seed = c(601L, 507L, 508L, 509L, 510L, 511L, 512L),
  dc_sA   = c("sA0p05_0p05", "sA0p1_0p1", "sA0p15_0p15", rep("sA0p05_0p05", 4)),
  dc_nu   = c("", "", "", "_nu0p05_0", "_nu-0p05_0", "_nu0_0p05", "_nu0_-0p05"),
  dc_seed = c(1001L, 1012L, 1013L, 1014L, 1015L, 1016L, 1017L), stringsAsFactors = FALSE)
nn_file <- function(sg, r) if (is.na(r$nn_tag))
  sprintf("nn1d_test/%s_nu0_c4w_e20k-30k-50k_tap2_s%d/results_e30000.rds", sg, r$nn_seed) else
  sprintf("nn1d_test/%s_%s_c4w_e30000_tap2_s%d/results.rds", sg, r$nn_tag, r$nn_seed)
dc_file <- function(sg, r) sprintf("runs_pois/pois_alt_%s_N2000_%s%s_lam1p3_K10_s%d_dmpnone.RData", sg, r$dc_sA, r$dc_nu, r$dc_seed)
power_1d <- function() do.call(rbind, lapply(seq_len(nrow(POWER_SETS)), function(i) {
  r <- POWER_SETS[i, ]
  do.call(rbind, lapply(c("NP1", "NP2", "NP3"), function(sg) rbind(
    cbind(witness = "NN", signal = sg, setting = r$setting, sigma_A = r$sigma_A, summ_z(read_rds_z(nn_file(sg, r), "Z_fit"))),
    cbind(witness = "dictionary", signal = sg, setting = r$setting, sigma_A = r$sigma_A,
          summ_z(read_rdata_z(dc_file(sg, r), "Z_fit_Q"))))))
}))
