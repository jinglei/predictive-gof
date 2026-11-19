## tex_tables_5d.R -- LaTeX validity table for Section 6 (dilepton benchmark), typeset from tables/table_5d.csv.
## Run from the repository root:  Rscript paper_epjc/scripts/tex_tables_5d.R  -> paper_epjc/tables/table_5d_null.tex
## Rates from 400 data sets are multiples of 0.0025, so four decimals are exact.
d <- read.csv("paper_epjc/tables/table_5d.csv", stringsAsFactors = FALSE)
nl <- d[d$setting == "null", ]
lev <- c("0" = "$0$", "+sigma_s" = "$+\\sigma_s$", "-sigma_s" = "$-\\sigma_s$", "+sigma_n" = "$+\\sigma_n$", "-sigma_n" = "$-\\sigma_n$")
nl <- nl[order(match(nl$nu_star, names(lev))), ]
rows <- sprintf("%s & $%.3f$ & %.3f & %.4f & %.4f \\\\", lev[nl$nu_star], nl$mean, nl$sd, nl$p05, nl$p01)
writeLines(c("\\begin{tabular}{lcccc}", "\\toprule",
  "$\\nu^*$ & mean of $Z$ & s.d.\\ of $Z$ & rate at $5\\%$ & rate at $1\\%$ \\\\", "\\midrule", rows,
  "\\bottomrule", "\\end{tabular}"), "paper_epjc/tables/table_5d_null.tex")
cat("-> paper_epjc/tables/table_5d_null.tex\n")
