## run_valid_5d.R -- 5-D NN null at the four off-centre nuisance points of NPLM (EPJC 82:275, Table tab:5D_validation):
## one nuisance shifted by +-1 sigma at a time, sigma_A = (0.025, 0.003) = (normalization, scale); headline cell
## c = 0.7, 3000 epochs, 400 toys, GPU duty 0.7.
##   plus   nu* = (0, +3e-3)    seed 111 (same toys as null_plus)    -> nn_test/sens_null_plus_c0.7/        [done 2026-10-16]
##   minus  nu* = (0, -3e-3)    seed 112 (same toys as null_minus)   -> nn_test/sens_null_minus_c0.7/       [done 2026-10-16]
##   nplus  nu* = (+0.025, 0)   seed 113                             -> nn_test/sens_null_nplus_c0.7/
##   nminus nu* = (-0.025, 0)   seed 114                             -> nn_test/sens_null_nminus_c0.7/
## Single epoch budget -> results.rds (not results_e3000.rds).
## run (from 5d/):  nohup nice -n 10 Rscript run_valid_5d.R nplus > run_valid_5d_nplus.log 2>&1 &
source("utils_5d.R"); source("nn_test_5d.R")
side <- match.arg(commandArgs(trailingOnly = TRUE)[1], c("plus", "minus", "nplus", "nminus"))
pt <- list(plus   = list(nu = c(0, 3e-3),  seed = 111L),  minus  = list(nu = c(0, -3e-3), seed = 112L),
           nplus  = list(nu = c(0.025, 0), seed = 113L),  nminus = list(nu = c(-0.025, 0), seed = 114L))[[side]]
dm <- load_delta_5d("delta/delta_raw5_p8.json")
sm <- load_sample_5d("DiLepton_SM"); zp <- load_sample_5d("Zprime300")
pools <- make_pools_5d(sm, list(Zprime300 = zp)); rm(sm, zp); invisible(gc())
run_nn_study_5d(pools, dm, B = 400, seed = pt$seed, nu_star = pt$nu, clip = 0.7, epochs = 3000L, duty = 0.7,
                cores = 8L, tag = sprintf("sens_null_%s_c0.7", side))
