## fig_1d_null.R -- 1-D null distributions at sigma_A = 0.05, five nuisance points (analogue of NPLM EPJC 82:275 Fig. 5),
## one figure per witness, 400 toys per point. Z with the fitting-variance term (NN: Z_fit; dictionary: Z_fit_Q) and
## without it (Z_orth / Z_orth_Q). Grey band: N(0, 1).
## Run from the repository root:  Rscript paper_figures_tables/scripts/fig_1d_null.R   -> paper_figures_tables/figures/fig_1d_null_{nn,dict}.pdf
source("paper_figures_tables/scripts/summ.R"); source("paper_figures_tables/scripts/figs.R")
lab <- list(c("0", "0"), c("+1", "0"), c("0", "+1"), c("-1", "0"), c("0", "-1"))   # (nu*_s/sigma_s, nu*_n/sigma_n)
## same order as lab; nu_star in code = c(nu_n, nu_s)
nn <- c("null_nu0_c4w_e20k-30k-50k_tap2_s501/results_e30000.rds", "null_nu0_0.05_c4w_e30000_tap2_s505/results.rds",
        "null_nu0.05_0_c4w_e30000_tap2_s503/results.rds", "null_nu0_-0.05_c4w_e30000_tap2_s506/results.rds",
        "null_nu-0.05_0_c4w_e30000_tap2_s504/results.rds")
dc <- c("pois_null_N2000_sA0p05_0p05_lam1p3_K10_s1002", "pois_null_N2000_sA0p05_0p05_nu0_0p05_lam1p3_K10_s1005",
        "pois_null_N2000_sA0p05_0p05_nu0p05_0_lam1p3_K10_s1003", "pois_null_N2000_sA0p05_0p05_nu0_-0p05_lam1p3_K10_s1006",
        "pois_null_N2000_sA0p05_0p05_nu-0p05_0_lam1p3_K10_s1004")
load_nn <- function(f) { x <- readRDS(file.path("nn1d_test", f)); if (!is.data.frame(x)) x <- x$results
  list(Z_fit = x$Z_fit, Z_orth = x$Z_orth) }
load_dc <- function(f) { e <- new.env(); load(sprintf("runs_pois/%s_dmpnone.RData", f), envir = e)
  list(Z_fit = e$results$Z_fit_Q, Z_orth = e$results$Z_orth_Q) }
build <- function(files, loader, out) {
  br <- seq(-4, 4, by = 0.4)
  pl <- lapply(seq_along(files), function(i) {
    z <- loader(files[i]); s <- summ_z(z$Z_fit); so <- summ_z(z$Z_orth)
    d <- rbind(data.frame(stat = "Z_fit", z = z$Z_fit), data.frame(stat = "Z_orth", z = z$Z_orth)); d <- d[is.finite(d$z), ]
    note <- sprintf("size 5%%: %.3f (%.3f without)\nsize 1%%: %.3f (%.3f without)", s$p05, so$p05, s$p01, so$p01)
    zhist(d, br, title = nu_title(lab[[i]][1], lab[[i]][2]), note = note)
  })
  p <- (pl[[1]] + ((pl[[2]] | pl[[3]]) / (pl[[4]] | pl[[5]])) + plot_layout(widths = c(1.15, 2), guides = "collect")) &
    theme(legend.position = "bottom")
  ggsave(out, p, width = 7.2, height = 4.4, device = cairo_pdf); cat("->", out, "\n")
}
build(nn, load_nn, "paper_figures_tables/figures/fig_1d_null_nn.pdf")
build(dc, load_dc, "paper_figures_tables/figures/fig_1d_null_dict.pdf")
