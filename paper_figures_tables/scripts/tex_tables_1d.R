## tex_tables_1d.R -- LaTeX tables for Section 5 (one-dimensional benchmark), typeset from the CSVs.
## Inputs: tables/table_1d.csv (table_1d.R), tables/table_1d_power.csv (table_1d_power.R).
## Run from the repository root:  Rscript paper_figures_tables/scripts/tex_tables_1d.R
##   -> paper_figures_tables/tables/table_1d_null.tex, paper_figures_tables/tables/table_1d_power.tex
## rates are multiples of 1/400 = 0.0025, so four decimals are exact
f3 <- function(x) ifelse(is.na(x), "--", sprintf("%.4f", round(x, 4)))
f2 <- function(x) sprintf("%.2f", round(x, 2))
d <- read.csv("paper_figures_tables/tables/table_1d.csv", stringsAsFactors = FALSE)
nu_lab <- c("0" = "$0$", "+sigma_n" = "$+\\sigma_n$", "-sigma_n" = "$-\\sigma_n$",
            "+sigma_s" = "$+\\sigma_s$", "-sigma_s" = "$-\\sigma_s$")
## ---- validity: sizes at 5% / 1% for both witnesses, every null setting ----
nl <- d[d$setting == "null", ]
keys <- unique(nl[, c("sigma_A", "nu_star")])
keys <- keys[order(keys$sigma_A, match(keys$nu_star, names(nu_lab))), ]
get <- function(w, sA, nu, col) { r <- nl[nl$witness == w & nl$sigma_A == sA & nl$nu_star == nu, col]; if (length(r)) r else NA }
rows <- apply(keys, 1, function(k) {
  sA <- as.numeric(k[["sigma_A"]]); nu <- trimws(k[["nu_star"]])
  sprintf("%.2f & %s & %s & %s & %s & %s \\\\", sA, nu_lab[[nu]],
          f3(get("NN", sA, nu, "p05")), f3(get("NN", sA, nu, "p01")),
          f3(get("dictionary", sA, nu, "p05")), f3(get("dictionary", sA, nu, "p01")))
})
writeLines(c("\\begin{tabular}{llcccc}", "\\toprule",
  " & & \\multicolumn{2}{c}{network} & \\multicolumn{2}{c}{dictionary} \\\\",
  "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
  "$\\sigma_A$ & $\\nu^*$ & $5\\%$ & $1\\%$ & $5\\%$ & $1\\%$ \\\\", "\\midrule", rows, "\\bottomrule", "\\end{tabular}"),
  "paper_figures_tables/tables/table_1d_null.tex")
## ---- power at sigma_A = 0.05, nu* = 0 ----
p <- read.csv("paper_figures_tables/tables/table_1d_power.csv", stringsAsFactors = FALSE)
p0 <- p[p$setting == "0" & p$sigma_A == 0.05, ]
prow <- function(sg) {
  a <- p0[p0$signal == sg & p0$witness == "NN", ]; b <- p0[p0$signal == sg & p0$witness == "dictionary", ]
  sprintf("%s & $%s \\pm %s$ & %s & %s & $%s \\pm %s$ & %s & %s \\\\", sg,
          f2(a$median), f2(a$se_median), f3(a$p05), f3(a$p01), f2(b$median), f2(b$se_median), f3(b$p05), f3(b$p01))
}
writeLines(c("\\begin{tabular}{lcccccc}", "\\toprule",
  " & \\multicolumn{3}{c}{network} & \\multicolumn{3}{c}{dictionary} \\\\",
  "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
  "signal & median $Z$ & $5\\%$ & $1\\%$ & median $Z$ & $5\\%$ & $1\\%$ \\\\", "\\midrule",
  sapply(c("NP1", "NP2", "NP3"), prow), "\\bottomrule", "\\end{tabular}"),
  "paper_figures_tables/tables/table_1d_power.tex")
cat("-> paper_figures_tables/tables/table_1d_null.tex, table_1d_power.tex\n")
