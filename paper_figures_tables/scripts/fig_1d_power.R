## fig_1d_power.R -- 1-D median Z for NP1-3 (analogue of NPLM EPJC 82:275 Fig. 8). Left: nu* = 0 against sigma_A
## (0.05, 0.10, 0.15). Right: sigma_A = 0.05 at nu* = 0 and the four one-at-a-time +-sigma points. Signal: colour,
## marker shape (circle / triangle / square) and, in the left panel, line style (solid / dashed / dotted), so the figure
## reads in black and white; witness: filled marker = network, open marker = dictionary. Bars: bootstrap se of the median (400 toys). Raw median Z, not divided by
## a model-dependent Z_ref: the +-sigma_s points move the background under the signal, so raw Z moves with them.
## Run from the repository root:  Rscript paper_figures_tables/scripts/fig_1d_power.R   -> paper_figures_tables/figures/fig_1d_power.pdf
source("paper_figures_tables/scripts/power_1d_data.R"); source("paper_figures_tables/scripts/figs.R")
SCOL <- c(NP1 = "#2a78d6", NP2 = "#eb6834", NP3 = "#1baf7a")   # validated: all pairs pass; aqua needs labels (relief)
## Signal: colour + shape + linetype (one merged legend key per signal). Witness: fill (signal colour = network,
## white = dictionary); the fill legend shows two keys only.
SSHP <- c(NP1 = 21, NP2 = 24, NP3 = 22)                 # circle / triangle / square (fillable)
SLTY <- c(NP1 = "solid", NP2 = "42", NP3 = "13")         # solid / dashed / dotted
WLAB <- c(NN = "network", dictionary = "dictionary")
FILL <- c(setNames(SCOL, paste0(names(SCOL), ".NN")), setNames(rep("white", 3), paste0(names(SCOL), ".dictionary")))
fill_scale <- scale_fill_manual(values = FILL, breaks = c("NP1.NN", "NP1.dictionary"), labels = unname(WLAB), name = NULL)
sig_guide <- guide_legend(order = 1, override.aes = list(fill = "white"))
fill_guide <- guide_legend(override.aes = list(shape = 21, colour = INK2, fill = c(INK2, "white"), linetype = 0), order = 2)
d <- power_1d(); d$witness <- factor(d$witness, c("NN", "dictionary"))
d$wf <- paste(d$signal, d$witness, sep = ".")
a <- d[d$setting == "0", ]
pa <- ggplot(a, aes(sigma_A, median, colour = signal, group = interaction(signal, witness))) +
  geom_line(aes(linetype = signal), linewidth = 0.6) +
  geom_errorbar(aes(ymin = median - se_median, ymax = median + se_median), width = 0.004, linewidth = 0.4) +
  geom_point(aes(shape = signal, fill = wf), size = 2.2, stroke = 0.8) +
  geom_text(data = a[a$sigma_A == 0.15 & a$witness == "NN", ], aes(label = signal), hjust = -0.35, size = 3,
            show.legend = FALSE, colour = INK) +
  scale_colour_manual(values = SCOL, name = NULL) +
  scale_shape_manual(values = SSHP, name = NULL) +
  scale_linetype_manual(values = SLTY, name = NULL) + fill_scale +
  scale_x_continuous(breaks = c(0.05, 0.10, 0.15), limits = c(0.04, 0.17)) +
  labs(x = expression(sigma[A]), y = "median Z", title = expression(nu^"*" == 0)) + theme_paper() +
  guides(colour = sig_guide, shape = sig_guide, linetype = sig_guide, fill = fill_guide)
lev <- c("0", "+sigma_n", "-sigma_n", "+sigma_s", "-sigma_s")
lab <- expression(0, "+"*sigma[n], "-"*sigma[n], "+"*sigma[s], "-"*sigma[s])
b <- d[d$sigma_A == 0.05, ]; b$x <- match(b$setting, lev) + ifelse(b$witness == "NN", -0.12, 0.12)
pb <- ggplot(b, aes(x, median, colour = signal)) +
  geom_errorbar(aes(ymin = median - se_median, ymax = median + se_median), width = 0.08, linewidth = 0.4) +
  geom_point(aes(shape = signal, fill = wf), size = 2.2, stroke = 0.8) +
  scale_colour_manual(values = SCOL, name = NULL) +
  scale_shape_manual(values = SSHP, name = NULL) + fill_scale +
  scale_x_continuous(breaks = seq_along(lev), labels = lab) +
  labs(x = "true nuisance shift", y = "median Z", title = expression(sigma[A] == 0.05)) + theme_paper() +
  guides(colour = "none", shape = "none", fill = "none")
p <- (pa | pb) + plot_layout(widths = c(1, 1.3), guides = "collect") & theme(legend.position = "bottom")
ggsave("paper_figures_tables/figures/fig_1d_power.pdf", p, width = 7.2, height = 3.4, device = cairo_pdf)
cat("-> paper_figures_tables/figures/fig_1d_power.pdf\n")
