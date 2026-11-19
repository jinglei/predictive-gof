## summ.R -- shared summary of a vector of Z values for the paper tables.
## Size/power at the nominal one-sided levels; median Z is the counterpart of NPLM's Z-bar.
## MC se: sd/sqrt(n) for the mean, binomial for rates, bootstrap (B resamples) for the median.
summ_z <- function(z, B = 2000L, seed = 1L) {
  z <- z[!is.na(z)]; n <- length(z); set.seed(seed)
  rate <- function(c) { p <- mean(z > c); c(p, sqrt(p * (1 - p) / n)) }
  r05 <- rate(qnorm(0.95)); r01 <- rate(qnorm(0.99))
  data.frame(n = n, mean = mean(z), se_mean = sd(z) / sqrt(n), sd = sd(z),
             median = median(z), se_median = sd(replicate(B, median(sample(z, replace = TRUE)))),
             p05 = r05[1], se_p05 = r05[2], p01 = r01[1], se_p01 = r01[2])
}
read_rds_z <- function(f, col) { x <- readRDS(f); if (!is.data.frame(x)) x <- x$results; x[[col]] }
read_rdata_z <- function(f, col) { e <- new.env(); load(f, envir = e); e$results[[col]] }
write_table <- function(tab, out) {
  write.csv(tab, out, row.names = FALSE)
  num <- vapply(tab, is.numeric, logical(1)); tab[num] <- lapply(tab[num], round, 3)
  print(tab, row.names = FALSE); cat("\n->", out, "\n")
}
