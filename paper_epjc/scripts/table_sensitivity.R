## table_sensitivity.R -- tuning grids (size and power per cell). See sensitivity_data.R for sources.
## Run from the repository root:  Rscript paper_epjc/scripts/table_sensitivity.R
##   -> paper_epjc/tables/table_sensitivity_1d.csv, paper_epjc/tables/table_sensitivity_5d.csv
source("paper_epjc/scripts/sensitivity_data.R")
write_table(rbind(sens_1d_nn(), sens_1d_dict()), "paper_epjc/tables/table_sensitivity_1d.csv")
write_table(sens_5d(), "paper_epjc/tables/table_sensitivity_5d.csv")
