## table_1d_power.R -- 1-D power grid: median Z and power at nominal 5% / 1%, NP1-3, NN and dictionary,
## nu* = 0 at sigma_A = 0.05 / 0.10 / 0.15 and the four +-sigma points at sigma_A = 0.05 (400 toys per row).
## Run from the repository root:  Rscript paper_figures_tables/scripts/table_1d_power.R   -> paper_figures_tables/tables/table_1d_power.csv
source("paper_figures_tables/scripts/power_1d_data.R")
write_table(power_1d(), "paper_figures_tables/tables/table_1d_power.csv")
