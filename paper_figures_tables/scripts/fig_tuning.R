## fig_tuning.R -- size and power across the tuning grids (NN clip x epochs), 1-D and 5-D. See sensitivity_data.R.
## First panel: null size at 5% (+- binomial se), dashed line at 0.05. Other panels: median Z (+- bootstrap se) per signal.
## Null panel: shaded Monte Carlo bands around the nominal 5% (+-1 se dark, +-2 se light; se from the number of
## null data sets per point). Clip values (smallest, middle, largest) are encoded three ways: colour (violet / red / green; validated with the
## dataviz validator on white: all >= 3:1 contrast; the red-green pair is at the CVD floor, legal only with secondary
## encoding), marker (circle / triangle / square) and line style (solid / dashed / dot-dash), so the figure reads in
## greyscale. Reference lines (grey long-dash, keyed in the legend): the nominal 5% level in the null panel (thin) and,
## in the 1-D signal panels, the dictionary at its default penalty (lambda_z = 1.3, damping "none"; thick).
## Run from the repository root:  Rscript paper_figures_tables/scripts/fig_tuning.R -> paper_figures_tables/figures/fig_{1d,5d}_tuning.pdf
source("paper_figures_tables/scripts/sensitivity_data.R"); source("paper_figures_tables/scripts/figs.R")
CCOL <- c("#6250d6", "#e34948", "#008300")             # violet, red, green
CSHP <- c(16, 17, 15)                                   # circle, triangle, square (filled)
CLTY <- c("solid", "22", "4212")                        # solid, dashed, dot-dash
panel <- function(d, sg, clips, ref = NULL, title = sg) {
  g <- d[d$setting == sg, ]; g$clip <- factor(g$clip, clips); null <- sg == "null"
  g$y <- if (null) g$p05 else g$median; g$se <- if (null) g$se_p05 else g$se_median
  ## training length as an evenly spaced category; the clip series are dodged into separate columns
  g$ep <- factor(g$epochs / 1000, levels = sort(unique(g$epochs)) / 1000)
  dodge <- position_dodge(width = 0.5)
  p <- ggplot(g, aes(ep, y, colour = clip, shape = clip, linetype = clip, group = clip))
  if (null) {
    ## Monte Carlo band for a valid test: nominal 0.05 +- 1 and 2 binomial standard errors at n null data sets
    se0 <- sqrt(0.05 * 0.95 / unique(g$n))
    stopifnot(length(se0) == 1L)
    p <- p + annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0.05 - 2 * se0, ymax = 0.05 + 2 * se0,
                      fill = "#f3f2ef", colour = NA) +
             annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0.05 - se0, ymax = 0.05 + se0,
                      fill = "#e6e4df", colour = NA)
  }
  p <- p +
    geom_line(linewidth = 0.6, position = dodge) +
    geom_errorbar(aes(ymin = y - se, ymax = y + se), width = 0.12, linewidth = 0.4, linetype = "solid",
                  position = dodge) +
    geom_point(size = 2, position = dodge) +
    scale_colour_manual(values = setNames(CCOL, clips), name = "clip") +
    scale_shape_manual(values = setNames(CSHP, clips), name = "clip") +
    scale_linetype_manual(values = setNames(CLTY, clips), name = "clip") +
    labs(x = "epochs (thousands)", y = if (null) "size at 5%" else "median Z", title = title) + theme_paper() +
    theme(legend.title = element_text(colour = INK), legend.key.width = unit(1.4, "lines"))
  ## reference lines as legend keys (alpha is used only to carry the key; it does not change the drawing)
  refs <- rbind(if (null) data.frame(y = 0.05, lab = "nominal level"),
                if (!is.null(ref)) data.frame(y = ref, lab = "dictionary (default penalty)"))
  if (!is.null(refs)) p <- p +
    geom_hline(data = refs, aes(yintercept = y, alpha = lab, linewidth = lab), linetype = "longdash", colour = REF) +
    scale_alpha_manual(values = c("nominal level" = 1, "dictionary (default penalty)" = 1), name = NULL) +
    scale_linewidth_manual(values = c("nominal level" = 0.5, "dictionary (default penalty)" = 1.1), name = NULL)
  p
}
d1 <- sens_1d_nn(); dd <- sens_1d_dict(); dh <- dd[dd$lambda_z == 1.3, ]
ref1 <- function(sg) if (sg == "null") NULL else dh$median[dh$setting == sg]
p1 <- (panel(d1, "null", c("3", "4", "6"), title = "null") | panel(d1, "NP1", c("3", "4", "6"), ref1("NP1")) |
       panel(d1, "NP2", c("3", "4", "6"), ref1("NP2")) | panel(d1, "NP3", c("3", "4", "6"), ref1("NP3"))) +
  plot_layout(guides = "collect") & theme(legend.position = "bottom")
ggsave("paper_figures_tables/figures/fig_1d_tuning.pdf", p1, width = 7.2, height = 2.9, device = cairo_pdf)
d5 <- sens_5d()
p5 <- (panel(d5, "null", c("0.7", "1", "1.3"), title = "null") | panel(d5, "Zprime300", c("0.7", "1", "1.3"), title = "Z′") |
       panel(d5, "EFT06_2", c("0.7", "1", "1.3"), title = "EFT")) +
  plot_layout(guides = "collect") & theme(legend.position = "bottom")
ggsave("paper_figures_tables/figures/fig_5d_tuning.pdf", p5, width = 6.2, height = 2.9, device = cairo_pdf)
cat("-> paper_figures_tables/figures/fig_1d_tuning.pdf, fig_5d_tuning.pdf\n")
