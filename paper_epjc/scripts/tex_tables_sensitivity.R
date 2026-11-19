## tex_tables_sensitivity.R -- LaTeX tables of the full tuning grids for Appendix D, typeset from
## tables/table_sensitivity_{1d,5d}.csv (<- table_sensitivity.R).
## Run from the repository root:  Rscript paper_epjc/scripts/tex_tables_sensitivity.R
##   -> paper_epjc/tables/table_sensitivity_{1d,5d}.tex
## Rates from 400 data sets are multiples of 0.0025: four decimals are exact (5-D table); the 1-D table rounds
## them half up to three. Median Z with its bootstrap
## standard error in parentheses (summ.R conventions). The default configuration is marked with a dagger.
med <- function(d, s) { r <- d[d$setting == s, ]; sprintf("%.2f (%.2f)", r$median, r$se_median) }
rate <- function(d, digits = 4) {               # digits < 4: round half up (ties at 0.0025 multiples)
  r <- d[d$setting == "null", ]; f <- function(x) sprintf(paste0("%.", digits, "f"), floor(x * 10^digits + 0.5 + 1e-9) / 10^digits)
  c(f(r$p05), f(r$p01)) }
epo <- function(e) sprintf("%s\\times10^4", format(e / 1e4))

## 1-D: network clip x epochs, dictionary lambda_z
d1 <- read.csv("paper_epjc/tables/table_sensitivity_1d.csv", stringsAsFactors = FALSE)
nn <- d1[d1$witness == "NN", ]; dc <- d1[d1$witness == "dictionary", ]
rows1 <- character(0)
for (c in sort(unique(nn$clip))) for (e in sort(unique(nn$epochs))) {
  g <- nn[nn$clip == c & nn$epochs == e, ]
  def <- if (c == 4 && e == 30000) "$^\\dagger$" else ""
  rows1 <- c(rows1, paste(c(sprintf("network & $(%g,%s)$%s", c, epo(e), def), rate(g, 3),
                            med(g, "NP1"), med(g, "NP2"), med(g, "NP3")), collapse = " & "))
}
for (l in sort(unique(dc$lambda_z))) {
  g <- dc[dc$lambda_z == l, ]
  def <- if (l == 1.3) "$^\\dagger$" else ""
  rows1 <- c(rows1, paste(c(sprintf("dictionary & $%g$%s", l, def), rate(g, 3),
                            med(g, "NP1"), med(g, "NP2"), med(g, "NP3")), collapse = " & "))
}
rows1 <- paste0(rows1, " \\\\")
i_dict <- which(grepl("^dictionary", rows1))[1]
rows1 <- append(rows1, "\\midrule", after = i_dict - 1)
writeLines(c("\\begin{tabular}{llccccc}", "\\toprule",
  " & setting & \\multicolumn{2}{c}{null: rate at} & \\multicolumn{3}{c}{median $Z$} \\\\",
  "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}",
  "witness & $(c,\\text{epochs})$ / $\\lambda$ & $5\\%$ & $1\\%$ & NP1 & NP2 & NP3 \\\\", "\\midrule", rows1,
  "\\bottomrule", "\\end{tabular}"), "paper_epjc/tables/table_sensitivity_1d.tex")
cat("-> paper_epjc/tables/table_sensitivity_1d.tex\n")

## 5-D: network clip x epochs
d5 <- read.csv("paper_epjc/tables/table_sensitivity_5d.csv", stringsAsFactors = FALSE)
rows5 <- character(0)
for (c in sort(unique(d5$clip))) for (e in sort(unique(d5$epochs))) {
  g <- d5[d5$clip == c & d5$epochs == e, ]
  def <- if (c == 0.7 && e == 3000) "$^\\dagger$" else ""
  vz <- sprintf("%.2f", g$sd[g$setting == "null"]^2)   # null variance of Z (sample s.d. squared)
  rows5 <- c(rows5, paste(c(sprintf("$%.1f$ & $%d$%s", c, e, def), vz, rate(g),
                            med(g, "Zprime300"), med(g, "EFT06_2")), collapse = " & "))
}
rows5 <- paste0(rows5, " \\\\")
writeLines(c("\\begin{tabular}{llccccc}", "\\toprule",
  " & & \\multicolumn{3}{c}{null} & \\multicolumn{2}{c}{median $Z$} \\\\",
  "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}",
  "$c$ & epochs & variance of $Z$ & rate at $5\\%$ & rate at $1\\%$ & $Z'$ & EFT \\\\", "\\midrule", rows5,
  "\\bottomrule", "\\end{tabular}"), "paper_epjc/tables/table_sensitivity_5d.tex")
cat("-> paper_epjc/tables/table_sensitivity_5d.tex\n")
