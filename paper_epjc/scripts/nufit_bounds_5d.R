## Pushforward nuisance fits at the bound of the search interval, 5-D studies in the paper.
## Reads the per-fold nu-hat saved in 5d/nn_test/<run>/toy_*.meta.rds (export_nn_batch_5d);
## a fit with no sign change of F_k is set to nu_s = +-bound (fit_null_push_5d), so the
## saved estimates identify it. The full-data fit (fold 0) is not saved.
## Output: paper_epjc/tables/nufit_bounds_5d.csv
## Usage (from the repository root): Rscript paper_epjc/scripts/nufit_bounds_5d.R
bound <- 0.01                                   # fit_null_push_5d / null_fits_push_5d default
runs <- c(sprintf("sens_%s_c%s", rep(c("null", "zp", "eft"), each = 3), c("0.7", "1", "1.3")),
          sprintf("sens_null_%s_c0.7", c("plus", "minus", "nplus", "nminus")))
root <- "5d/nn_test"
tab <- do.call(rbind, lapply(runs, function(r) {
  f <- list.files(file.path(root, r), "^toy_[0-9]+\\.meta\\.rds$", full.names = TRUE)
  s <- unlist(lapply(f, function(x) vapply(readRDS(x)$folds, function(z) z$nu[2], 0)))
  data.frame(run = r, toys = length(f), fold_fits = length(s),
             at_bound = sum(abs(s) >= bound - 1e-12), max_abs_nu_s = max(abs(s)))
}))
out <- "paper_epjc/tables/nufit_bounds_5d.csv"
if (file.exists(out)) stop("refusing to overwrite ", out)
write.csv(tab, out, row.names = FALSE)
print(tab)
