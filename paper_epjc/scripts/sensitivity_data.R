## sensitivity_data.R -- tuning grids, read from saved outputs. 400 null toys per cell (1-D signals 400; 5-D signals 160).
##   1-D NN:   clip {3, 4, 6} (weights only) x epochs {20k, 30k, 50k}, taper kappa = 2, sigma_A = 0.05, nu* = 0;
##             null seed 501, NP1-3 seed 601 (run_nn1d_grid.R <clip>); nn1d_test/<sig>_nu0_c<clip>w_e20k-30k-50k_tap2_s*/results_e*.rds
##   1-D dict: lambda_z {1.1, 1.3, 1.5}, damping "none", paired across lambda_z (null seed 1002, NP1-3 seed 1001, same toys);
##             runs_pois/*_lam<l>_K10_s*_dmpnone.RData <- launch_pois_dmpnone.R (1.3), run_lambda_scan_1d.R (1.1, 1.5).
##             The soft-damping runs (no _dmpnone suffix) are superseded and not read.
##   5-D NN:   clip {0.7, 1.0, 1.3} (all parameters) x epochs {2000, 3000, 5000}; null nu* = 0 (seed 201),
##             Z' and EFT at nu*_s = +sigma_s (seeds 61, 71); 5d/nn_test/sens_<set>_c<clip>/results_e*.rds (5d/run_nn_sens_5d.R)
source("paper_epjc/scripts/summ.R")
cell <- function(tag) function(z) summ_z(z)
sens_1d_nn <- function() do.call(rbind, lapply(c(3, 4, 6), function(cl) do.call(rbind, lapply(c(20000, 30000, 50000), function(ep)
  do.call(rbind, lapply(c("null", "NP1", "NP2", "NP3"), function(sg) {
    f <- sprintf("nn1d_test/%s_nu0_c%dw_e20k-30k-50k_tap2_s%d/results_e%d.rds", sg, cl, if (sg == "null") 501L else 601L, ep)
    cbind(witness = "NN", setting = sg, clip = cl, epochs = ep, lambda_z = NA, damping = NA, summ_z(read_rds_z(f, "Z_fit")))
  }))))))
sens_1d_dict <- function() {
  runs <- list(list(1.1, "none", "lam1p1", "_dmpnone"), list(1.3, "none", "lam1p3", "_dmpnone"),
               list(1.5, "none", "lam1p5", "_dmpnone"))
  do.call(rbind, lapply(runs, function(r) do.call(rbind, lapply(c("null", "NP1", "NP2", "NP3"), function(sg) {
    stem <- if (sg == "null") "pois_null" else paste0("pois_alt_", sg)
    f <- sprintf("runs_pois/%s_N2000_sA0p05_0p05_%s_K10_s%d%s.RData", stem, r[[3]], if (sg == "null") 1002L else 1001L, r[[4]])
    cbind(witness = "dictionary", setting = sg, clip = NA, epochs = NA, lambda_z = r[[1]], damping = r[[2]],
          summ_z(read_rdata_z(f, "Z_fit_Q")))
  }))))
}
sens_5d <- function() do.call(rbind, lapply(c(0.7, 1, 1.3), function(cl) do.call(rbind, lapply(c(2000, 3000, 5000), function(ep)
  do.call(rbind, lapply(c(null = "null", Zprime300 = "zp", EFT06_2 = "eft"), function(st) {
    f <- sprintf("5d/nn_test/sens_%s_c%g/results_e%d.rds", st, cl, ep)
    cbind(witness = "NN", setting = names(which(c(null = "null", Zprime300 = "zp", EFT06_2 = "eft") == st)),
          clip = cl, epochs = ep, summ_z(read_rds_z(f, "Z_fit")))
  }))))))
