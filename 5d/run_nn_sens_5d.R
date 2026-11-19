source("utils_5d.R"); source("nn_test_5d.R")
dm <- load_delta_5d("delta/delta_raw5_p8.json")
sm <- load_sample_5d("DiLepton_SM"); zp <- load_sample_5d("Zprime300"); eft <- load_sample_5d("EFT06_2")
pools <- make_pools_5d(sm, list(Zprime300 = zp, EFT06_2 = eft)); rm(sm, zp, eft); invisible(gc())
EP <- c(2000, 3000, 5000); runs <- list()
settings <- list(
  null = list(B = 400, seed = 201, args = list(nu_star = c(0, 0))),
  zp   = list(B = 160, seed = 61,  args = list(signal = "Zprime300", NS = 120, nu_star = c(0, 3e-3))),
  eft  = list(B = 160, seed = 71,  args = list(data = "EFT06_2", N_data = 1.0216 * CFG5$N0, nu_star = c(0, 3e-3))))
for (st in names(settings)) for (c in c(0.7, 1.0, 1.3)) {
  s <- settings[[st]]; tg <- sprintf("sens_%s_c%g", st, c)
  do.call(run_nn_study_5d, c(list(pools = pools, dm = dm, B = s$B, seed = s$seed, clip = c, epochs = EP, tag = tg), s$args))
  runs[[length(runs) + 1]] <- list(setting = st, clip = c, tag = tg)
  saveRDS(runs, "nn_test/sens_runs.rds")
}
cat("\n==== sensitivity table ====\n"); sens_table_5d("nn_test", runs, EP)
