## table_5d.R -- 5-D table: NN witness, headline c = 0.7, 3000 epochs.
## Nulls at nu* = 0 (seed 201; 5d/run_nn_sens_5d.R); scale +-sigma (seeds 111, 112) and normalization +-sigma_n
## (seeds 113, 114; 5d/run_valid_5d.R);
## 400 toys each; Z' N_S = 120 (seed 61) and EFT06_2 (seed 71) at nu* = +sigma, 160 toys (5d/run_nn_sens_5d.R).
## Run from the repository root:  Rscript paper_epjc/scripts/table_5d.R   -> paper_epjc/tables/table_5d.csv
source("paper_epjc/scripts/summ.R")
d <- "5d/nn_test/"
rows <- list(list("null",      "0",      "sens_null_c0.7/results_e3000.rds"),
             list("null",      "+sigma_s", "sens_null_plus_c0.7/results.rds"),
             list("null",      "-sigma_s", "sens_null_minus_c0.7/results.rds"),
             list("null",      "+sigma_n", "sens_null_nplus_c0.7/results.rds"),
             list("null",      "-sigma_n", "sens_null_nminus_c0.7/results.rds"),
             list("Zprime300", "+sigma_s", "sens_zp_c0.7/results_e3000.rds"),
             list("EFT06_2",   "+sigma_s", "sens_eft_c0.7/results_e3000.rds"))
tab <- do.call(rbind, lapply(rows, function(r)
  cbind(setting = r[[1]], nu_star = r[[2]], summ_z(read_rds_z(paste0(d, r[[3]]), "Z_fit")))))
write_table(tab, "paper_epjc/tables/table_5d.csv")
