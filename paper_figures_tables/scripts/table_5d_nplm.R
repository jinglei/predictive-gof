## table_5d_nplm.R -- 5-D sensitivity vs NPLM, electron-like regime (table version of NPLM EPJC 82:275 Fig. 25).
## Ours: NN headline c = 0.7, 3000 epochs, nu*_s = +sigma_s, 160 toys. NPLM: printed on
## Figures/5D_sensitivity_to_NP_elelike_v3b.pdf (electron-like, nu*_s = +sigma_s, 100 toys):
## Z' Z-bar = 3.9 +- 0.2, Z-bar_ref = 10.5 +- 0.1; EFT Z-bar = 5.1 +- 0.3, Z-bar_ref = 14.3 +- 0.2.
## Run from the repository root:  Rscript paper_figures_tables/scripts/table_5d_nplm.R   -> paper_figures_tables/tables/table_5d_nplm.{csv,tex}
source("paper_figures_tables/scripts/summ.R")
nplm <- data.frame(signal = c("Zprime300", "EFT06_2"), dir = c("sens_zp_c0.7", "sens_eft_c0.7"),
                   nplm_Z = c(3.9, 5.1), nplm_se = c(0.2, 0.3), Zref = c(10.5, 14.3), Zref_se = c(0.1, 0.2))
tab <- do.call(rbind, lapply(seq_len(nrow(nplm)), function(i) {
  r <- nplm[i, ]; s <- summ_z(read_rds_z(sprintf("5d/nn_test/%s/results_e3000.rds", r$dir), "Z_fit"))
  data.frame(signal = r$signal, toys = s$n, median_Z = s$median, se_median = s$se_median, p05 = s$p05, p01 = s$p01,
             nplm_Z = r$nplm_Z, nplm_se = r$nplm_se, Zref = r$Zref,
             ratio_ours = s$median / r$Zref, ratio_nplm = r$nplm_Z / r$Zref)
}))
write_table(tab, "paper_figures_tables/tables/table_5d_nplm.csv")
f2 <- function(v) sprintf("%.2f", v)
## Z-bar_ref ratio columns kept in the csv only; the paper table compares with NPLM directly (2026-11-19).
lines <- c("\\begin{tabular}{lcccc}", "\\toprule",
  " & \\multicolumn{3}{c}{this work (NN witness)} & NPLM \\\\",
  "signal & median $Z$ & power 5\\% & power 1\\% & $\\overline{Z}$ \\\\",
  "\\midrule",
  sprintf("%s & $%s \\pm %s$ & %s & %s & $%s \\pm %s$ \\\\", c("$Z'$", "EFT"), f2(tab$median_Z),
          f2(tab$se_median), f2(tab$p05), f2(tab$p01), sprintf("%.1f", tab$nplm_Z), sprintf("%.1f", tab$nplm_se)),
  "\\bottomrule", "\\end{tabular}")
writeLines(lines, "paper_figures_tables/tables/table_5d_nplm.tex"); cat("-> paper_figures_tables/tables/table_5d_nplm.tex\n")
