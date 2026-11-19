## table_1d.R -- 1-D table: NN and dictionary witnesses, Poisson mode, N0 = 2000, K = 10, 400 toys per row.
## Power rows (NP1-3) at nu* = 0, sigma_A = 0.05. Null rows at nu* = 0 and at NPLM's one-at-a-time points
## (+-sigma on nu_n or nu_s): both witnesses at sigma_A = 0.05; dictionary also at sigma_A = 0.15;
## nu* = 0 nulls at sigma_A = 0.10 (both); NN at all five points for sigma_A = 0.15 (run_valid_1d_nn015.R). Power grid over sigma_A / nu*: table_1d_power.R.
## Run from the repository root:  Rscript paper_figures_tables/scripts/table_1d.R   -> paper_figures_tables/tables/table_1d.csv
source("paper_figures_tables/scripts/summ.R")
nn_f  <- function(tag) file.path("nn1d_test", tag)
dic_f <- function(stem) sprintf("runs_pois/%s_dmpnone.RData", stem)
nn_null <- c(`0` = "null_nu0_c4w_e20k-30k-50k_tap2_s501/results_e30000.rds",          # run_nn1d_grid.R 4
             `+sigma_n` = "null_nu0.05_0_c4w_e30000_tap2_s503/results.rds",            # run_valid_1d.R nn
             `-sigma_n` = "null_nu-0.05_0_c4w_e30000_tap2_s504/results.rds",
             `+sigma_s` = "null_nu0_0.05_c4w_e30000_tap2_s505/results.rds",
             `-sigma_s` = "null_nu0_-0.05_c4w_e30000_tap2_s506/results.rds")
nn_null_sA <- list(list(0.10, "0", "null_nu0_c4w_e30000_tap2_s513/results.rds"),            # run_power_1d.R nn
                   list(0.15, "0", "null_nu0_c4w_e30000_tap2_s514/results.rds"),
                   list(0.15, "+sigma_n", "null_nu0.15_0_c4w_e30000_tap2_s515/results.rds"),    # run_valid_1d_nn015.R
                   list(0.15, "-sigma_n", "null_nu-0.15_0_c4w_e30000_tap2_s516/results.rds"),
                   list(0.15, "+sigma_s", "null_nu0_0.15_c4w_e30000_tap2_s517/results.rds"),
                   list(0.15, "-sigma_s", "null_nu0_-0.15_c4w_e30000_tap2_s518/results.rds"))
dic_null <- list(                                                                     # launch_pois_dmpnone.R; run_valid_1d_dict.R
  list(0.05, "0",        "pois_null_N2000_sA0p05_0p05_lam1p3_K10_s1002"),
  list(0.05, "+sigma_n", "pois_null_N2000_sA0p05_0p05_nu0p05_0_lam1p3_K10_s1003"),
  list(0.05, "-sigma_n", "pois_null_N2000_sA0p05_0p05_nu-0p05_0_lam1p3_K10_s1004"),
  list(0.05, "+sigma_s", "pois_null_N2000_sA0p05_0p05_nu0_0p05_lam1p3_K10_s1005"),
  list(0.05, "-sigma_s", "pois_null_N2000_sA0p05_0p05_nu0_-0p05_lam1p3_K10_s1006"),
  list(0.10, "0",        "pois_null_N2000_sA0p1_0p1_lam1p3_K10_s1018"),                    # run_power_1d.R dict
  list(0.15, "0",        "pois_null_N2000_sA0p15_0p15_lam1p3_K10_s1007"),
  list(0.15, "+sigma_n", "pois_null_N2000_sA0p15_0p15_nu0p15_0_lam1p3_K10_s1008"),
  list(0.15, "-sigma_n", "pois_null_N2000_sA0p15_0p15_nu-0p15_0_lam1p3_K10_s1009"),
  list(0.15, "+sigma_s", "pois_null_N2000_sA0p15_0p15_nu0_0p15_lam1p3_K10_s1010"),
  list(0.15, "-sigma_s", "pois_null_N2000_sA0p15_0p15_nu0_-0p15_lam1p3_K10_s1011"))
row <- function(w, set, sA, nu, z) cbind(witness = w, setting = set, sigma_A = sA, nu_star = nu, summ_z(z))
tab <- rbind(
  do.call(rbind, lapply(names(nn_null), function(k) row("NN", "null", 0.05, k, read_rds_z(nn_f(nn_null[[k]]), "Z_fit")))),
  do.call(rbind, lapply(nn_null_sA, function(r) row("NN", "null", r[[1]], r[[2]], read_rds_z(nn_f(r[[3]]), "Z_fit")))),
  do.call(rbind, lapply(c("NP1", "NP2", "NP3"), function(sg)
    row("NN", sg, 0.05, "0", read_rds_z(nn_f(sprintf("%s_nu0_c4w_e20k-30k-50k_tap2_s601/results_e30000.rds", sg)), "Z_fit")))),
  do.call(rbind, lapply(dic_null, function(r) row("dictionary", "null", r[[1]], r[[2]], read_rdata_z(dic_f(r[[3]]), "Z_fit_Q")))),
  do.call(rbind, lapply(c("NP1", "NP2", "NP3"), function(sg)
    row("dictionary", sg, 0.05, "0", read_rdata_z(dic_f(sprintf("pois_alt_%s_N2000_sA0p05_0p05_lam1p3_K10_s1001", sg)), "Z_fit_Q")))))
write_table(tab, "paper_figures_tables/tables/table_1d.csv")
