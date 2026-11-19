## fig_5d_null.R -- 5-D NN null distributions at the five nuisance points (analogue of NPLM EPJC 82:275 Fig. 16,
## electron-like). Headline c = 0.7, 3000 epochs, 400 toys per point. Grey band: N(0, 1).
## Run from the repository root:  Rscript paper_epjc/scripts/fig_5d_null.R   -> paper_epjc/figures/fig_5d_null.pdf
source("paper_epjc/scripts/summ.R"); source("paper_epjc/scripts/figs.R")
d <- "5d/nn_test/"
pts <- list(list("sens_null_c0.7/results_e3000.rds", "0", "0"),
            list("sens_null_plus_c0.7/results.rds",  "+1", "0"), list("sens_null_nplus_c0.7/results.rds",  "0", "+1"),
            list("sens_null_minus_c0.7/results.rds", "-1", "0"), list("sens_null_nminus_c0.7/results.rds", "0", "-1"))
br <- seq(-4, 4, by = 0.4)
pl <- lapply(pts, function(pt) {
  x <- readRDS(paste0(d, pt[[1]])); if (!is.data.frame(x)) x <- x$results
  s <- summ_z(x$Z_fit); so <- summ_z(x$Z_orth)
  note <- sprintf("size 5%%: %.3f (%.3f without)\nsize 1%%: %.3f (%.3f without)", s$p05, so$p05, s$p01, so$p01)
  zhist(zlong(x), br, title = nu_title(pt[[2]], pt[[3]]), note = note)
})
p <- (pl[[1]] + ((pl[[2]] | pl[[3]]) / (pl[[4]] | pl[[5]])) + plot_layout(widths = c(1.15, 2), guides = "collect")) &
  theme(legend.position = "bottom")
ggsave("paper_epjc/figures/fig_5d_null.pdf", p, width = 7.2, height = 4.4, device = cairo_pdf)
cat("-> paper_epjc/figures/fig_5d_null.pdf\n")
