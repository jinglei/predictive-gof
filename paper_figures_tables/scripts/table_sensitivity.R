## table_sensitivity.R -- tuning grids (size and power per cell). See sensitivity_data.R for sources.
## Run from the repository root:  Rscript paper_figures_tables/scripts/table_sensitivity.R
##   -> paper_figures_tables/tables/table_sensitivity_1d.csv, paper_figures_tables/tables/table_sensitivity_5d.csv
source("paper_figures_tables/scripts/sensitivity_data.R")
write_table(rbind(sens_1d_nn(), sens_1d_dict()), "paper_figures_tables/tables/table_sensitivity_1d.csv")
write_table(sens_5d(), "paper_figures_tables/tables/table_sensitivity_5d.csv")
