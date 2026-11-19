## figs.R -- shared plotting helpers for the paper figures (ggplot2 + patchwork).
## Two series per panel, as NPLM's tau / t pairs: Z_fit (with the fitting-variance correction V_cross)
## and Z_orth (without it). Palette validated (dataviz validator, white surface, all pairs pass).
suppressPackageStartupMessages({ library(ggplot2); library(patchwork) })
COL <- c(Z_fit = "#2a78d6", Z_orth = "#eb6834"); REF <- "#8a8984"; INK <- "#0b0b0b"; INK2 <- "#52514e"
LAB <- c(Z_fit = "Z (with fitting-variance term)", Z_orth = "Z (without)")
theme_paper <- function(base = 9) theme_minimal(base_size = base) %+replace% theme(
  panel.grid.minor = element_blank(), panel.grid.major = element_line(colour = "#ebeae6", linewidth = 0.25),
  axis.line = element_line(colour = INK2, linewidth = 0.3), axis.ticks = element_line(colour = INK2, linewidth = 0.3),
  axis.text = element_text(colour = INK2), axis.title = element_text(colour = INK),
  legend.position = "top", legend.title = element_blank(), legend.text = element_text(colour = INK),
  plot.title = element_text(colour = INK, size = base, hjust = 0, margin = margin(b = 2)),
  plot.subtitle = element_text(colour = INK2, size = base - 1.5, hjust = 0, lineheight = 1.05, margin = margin(b = 4)),
  plot.background = element_rect(fill = "white", colour = NA))
## long data: one row per toy and statistic
zlong <- function(x, stats = c("Z_fit", "Z_orth"))
  do.call(rbind, lapply(stats, function(s) data.frame(stat = s, z = x[[s]][is.finite(x[[s]])])))
## step-outline density histogram per statistic, with an N(0, 1) reference curve
zhist <- function(d, breaks, title = NULL, ref = TRUE, note = NULL) {
  h <- do.call(rbind, lapply(split(d, d$stat), function(g) {
    cnt <- hist(pmin(pmax(g$z, min(breaks)), max(breaks) - 1e-9), breaks = breaks, plot = FALSE)$counts
    data.frame(stat = g$stat[1], x = breaks, y = c(cnt, 0) / (nrow(g) * diff(breaks)[1])) }))
  p <- ggplot(h, aes(x, y, colour = stat)) + geom_step(linewidth = 0.6, direction = "hv")
  if (ref) p <- p + stat_function(fun = dnorm, inherit.aes = FALSE, colour = REF, linewidth = 1.6, alpha = 0.45,
                                  xlim = range(breaks), n = 400)
  p <- p + scale_colour_manual(values = COL, labels = LAB, breaks = names(COL)) +
    scale_x_continuous(expand = c(0, 0)) + scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
    labs(x = "Z", y = "density", title = title, subtitle = note) + theme_paper()
  p
}
## nuisance-point title: (nu*_s / sigma_s, nu*_n / sigma_n) = (a, b)
nu_title <- function(a, b) bquote("(" * nu[s]^"*" / sigma[s] * ", " * nu[n]^"*" / sigma[n] * ") = (" * .(a) * ", " * .(b) * ")")
