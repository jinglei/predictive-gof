source("utils.R"); source("nn1d_test.R"); source("nn1d_trunc.R")
cl <- as.numeric(commandArgs(trailingOnly = TRUE)[1]); EP <- c(20000, 30000, 50000)
for (st in c("null", "NP1", "NP2", "NP3"))
  run_nn1d_study(B = 400, seed = if (st == "null") 501 else 601, sig = if (st == "null") "NP1" else st,
                 signal = st != "null", clip = cl, bias_clip = FALSE, epochs = EP, kappa = 2,
                 batch = 100, cores = 8, keep_outputs = FALSE)
