## fig_5d_power.R -- 5-D NN Z distributions under the alternatives (analogue of NPLM EPJC 82:275 Fig. 23,
## column (b), electron-like): Z' (m = 300 GeV, N_S = 120) and EFT (c_W = 1 TeV^-2), nu*_s = +sigma_s,
## headline c = 0.7, 3000 epochs, 160 toys each. Grey band: N(0, 1), the null reference.
## NPLM values quoted from Figures/5D_sensitivity_to_NP_elelike_v3b.pdf (visible layer; 100 toys).
## Run from the repository root:  Rscript paper_epjc/scripts/fig_5d_power.R   -> paper_epjc/figures/fig_5d_power.pdf
source("paper_epjc/scripts/summ.R"); source("paper_epjc/scripts/figs.R")
sets <- list(list("sens_zp_c0.7",  "Z′ scenario (m = 300 GeV, N(S) = 120)", "3.9 ± 0.2"),
             list("sens_eft_c0.7", quote("EFT scenario (" * c[W] == "1.0 TeV"^-2 * ")"),   "5.1 ± 0.3"))
br <- seq(-3, 12, by = 0.5)
pl <- lapply(seq_along(sets), function(i) {
  x <- readRDS(sprintf("5d/nn_test/%s/results_e3000.rds", sets[[i]][[1]])); if (!is.data.frame(x)) x <- x$results
  s <- summ_z(x$Z_fit)
  note <- sprintf("median Z = %.2f ± %.2f;  power 5%% / 1%%: %.2f / %.2f\nNPLM: median Z = %s", s$median,
                  s$se_median, s$p05, s$p01, sets[[i]][[3]])
  zhist(zlong(x), br, title = sets[[i]][[2]], note = note)
})
p <- (pl[[1]] / pl[[2]] + plot_layout(guides = "collect")) & theme(legend.position = "top")
ggsave("paper_epjc/figures/fig_5d_power.pdf", p, width = 4.6, height = 5.4, device = cairo_pdf)
cat("-> paper_epjc/figures/fig_5d_power.pdf\n")
