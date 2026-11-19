## =============================================================
## nn1d_trunc.R -- re-assemble a finished 1-D NN-witness study with the witness restricted to
## each fold's training range:  phi_k(x) = f_k(x) * 1{x <= c_k},  c_k = the m-th largest
## training event of fold k (m_edge = 1: the training maximum; m_edge = 0: no truncation).
## c_k is a function of fold k's training data only, so phi_k is still a valid fold witness.
## No retraining: the networks' parameters are read from toy_XXXX.out.bin and evaluated in R.
## Taper variant (kappa = number): instead of the cut, a smooth weight from the fold's fitted null only,
##   tau_k(x) = min{1, Lambda_tr,k (1 - F_nu-hat,k(x)) / kappa},
## the expected number of TRAINING events beyond x under the fold's fitted null, over kappa: the witness
## is fully on while at least kappa training events are expected beyond x, then fades to 0. tau_k depends
## on fold k's training data only (through nu-hat), not on the network, so phi_k = tau_k f_k is valid and
## the mixed-Gram V_cross below applies with J~_k = tau_k J_k.
## Requires source("utils.R"); source("nn1d_test.R").
##
## Fitting randomness with a truncated witness. The parameters respond to the training data
## through the FULL network (training gradient J_k), the statistic through the TRUNCATED one
## (J~_k = J_k 1{x <= c_k}). To first order
##   Cov(S_j, S_k) = E_j E_k tr( D_k H_k^-1 M_kj D_j H_j^-1 M_jk ),
##   H_k = J_k' diag(mu_k) J_k,   M_kj = J_k' diag(wp) J~_j,   M_jk = J_j' diag(wp) J~_k,
## which reduces to the untruncated rule (M_jk = M_kj') when c = Inf. Free parameters, damping
## (output bias 1 - rho_n) and the ridge as in nn1d_test.R.
## =============================================================

.nn_eval <- function(th, x, h) {                      # theta = [W1 (h), b1 (h), W2 (h), b2]
  W1 <- th[1:h]; b1 <- th[h + 1:h]; W2 <- th[2 * h + 1:h]; b2 <- th[3 * h + 1]
  n <- length(x)
  s  <- plogis(outer(x, W1) + matrix(b1, n, h, byrow = TRUE))
  ds <- s * (1 - s) * matrix(W2, n, h, byrow = TRUE)
  list(f = as.vector(s %*% W2) + b2, J = cbind(ds * x, ds, s, 1))
}

.nn1d_tau <- function(x, fr, cutk, kappa) {
  if (!is.null(kappa)) pmin(1, fr$expo_tr * pexp(x, rate = fr$theta, lower.tail = FALSE) / kappa)
  else as.numeric(x <= cutk)
}

assemble_nn1d_trunc <- function(b, dir, m_edge, sigma_A, N0, K, n_grid, ridge = 1e-4, scale = 1, kappa = NULL,
                                suffix = "") {
  m  <- readRDS(file.path(dir, sprintf("toy_%04d.meta.rds", b)))
  o  <- .read_nn1d_out(file.path(dir, sprintf("toy_%04d%s.out.bin", b, suffix)))
  fg <- .nn1d_frames(m$x, m$folds, m$nu_A, sigma_A, N0, K, n_grid)
  P <- o$P; h <- (P - 1L) %/% 3L
  xr <- fg[[1]]$g$x_grid; wr <- fg[[1]]$g$wp
  pf <- Jf <- Jt <- mu <- vector("list", K); cut <- numeric(K); tau_ev <- numeric(m$N)
  for (k in seq_len(K)) {
    th  <- o$folds[[k]]$theta; idx <- which(m$folds == k); g <- fg[[k]]$g
    cut[k] <- if (m_edge > 0) sort(m$x[m$folds != k], decreasing = TRUE)[m_edge] else Inf
    fk <- fg[[k]]$fr
    ev <- .nn_eval(th, m$x[idx] / scale, h); gg <- .nn_eval(th, g$x_grid / scale, h); rr <- .nn_eval(th, xr / scale, h)
    te <- .nn1d_tau(m$x[idx], fk, cut[k], kappa); tau_ev[idx] <- te
    pf[[k]] <- make_fold_record(fk, g$x_grid, g$wp, idx, m$x[idx],
                                ev$f * te, gg$f * .nn1d_tau(g$x_grid, fk, cut[k], kappa))
    Jf[[k]] <- rr$J; Jt[[k]] <- rr$J * .nn1d_tau(xr, fk, cut[k], kappa)
    mu[[k]] <- fg[[k]]$fr$expo_tr * wr * exp(pmin(rr$f, 50))
  }
  r <- cvgof_test_quad(list(per_fold = pf, n = m$N, mode = "pois"))
  expo_te <- vapply(fg, function(z) z$fr$expo_te, 0); rho_n <- vapply(fg, function(z) z$fr$shrink[["nu_n"]], 0)
  fr <- lapply(o$folds, `[[`, "free")
  Hinv <- lapply(seq_len(K), function(k) {
    Jk <- Jf[[k]][, fr[[k]], drop = FALSE]; H <- crossprod(Jk, Jk * mu[[k]])
    H <- H + ridge * mean(diag(H)) * diag(nrow(H))
    ifelse(which(fr[[k]]) == P, 1 - rho_n[k], 1) * solve(H)
  })
  Vc <- 0
  for (j in seq_len(K - 1L)) for (k in (j + 1L):K) {
    Mkj <- crossprod(Jf[[k]][, fr[[k]], drop = FALSE], Jt[[j]][, fr[[j]], drop = FALSE] * wr)
    Mjk <- crossprod(Jf[[j]][, fr[[j]], drop = FALSE], Jt[[k]][, fr[[k]], drop = FALSE] * wr)
    Vc  <- Vc + 2 * expo_te[j] * expo_te[k] * sum((Hinv[[k]] %*% Mkj) * t(Hinv[[j]] %*% Mjk))
  }
  S <- r$orth$S; VQ <- r$orth$V_q; VX <- r$orth$V_x; va <- r$v_aux
  data.frame(toy = b, n_sig = m$n_sig, N = m$N, S = S, V_Q = VQ, V_X = VX, v_aux = va,
             V_cross = Vc, V_cross0 = NA_real_, n_free = mean(vapply(fr, sum, 0)),
             Z_fit = S / sqrt(VQ + va + Vc), Z_fit_X = S / sqrt(VX + va + Vc), Z_fit0 = NA_real_,
             Z_orth = r$orth$Z_Q, Z_raw = r$raw$Z_Q,
             ev_cut = sum(tau_ev < 1),                         # held-out events shrunk (cut: beyond their fold's cut)
             wt_removed = sum(1 - tau_ev))                     # held-out event weight removed
}

## re-assemble every toy of a finished study directory with one m_edge; saves results_trunc<m>.rds
reassemble_nn1d_trunc <- function(dir, m_edge = 0, cores = 8L, ridge = NULL, kappa = NULL) {
  dir <- path.expand(dir); R0 <- readRDS(file.path(dir, "results.rds")); mt <- attr(R0, "meta")
  rg  <- if (is.null(ridge)) mt$ridge else ridge
  rr  <- parallel::mclapply(R0$toy, function(b) try(assemble_nn1d_trunc(b, dir, m_edge, mt$sigma_A, mt$N0, mt$K,
                                                                         4001L, rg, kappa = kappa)), mc.cores = cores)
  bad <- vapply(rr, inherits, logical(1), "try-error"); if (any(bad)) stop("toy failed: ", rr[[which(bad)[1]]])
  R <- do.call(rbind, rr)
  lab <- if (is.null(kappa)) sprintf("witness cut at training edge, m = %d", m_edge) else sprintf("null-tail taper, kappa = %g", kappa)
  mt$tag <- sprintf("%s [%s]", mt$tag, lab); mt$ridge <- rg
  attr(R, "meta") <- mt
  saveRDS(R, file.path(dir, if (is.null(kappa)) sprintf("results_trunc%d.rds", m_edge) else sprintf("results_taper%g.rds", kappa)))
  summary_nn1d(R)
  cat(sprintf("  cor(S, V_Q) %.2f, cor(S, V_X) %.2f;  skewness of S %.2f;  held-out events shrunk %.2f per toy (weight removed %.2f)\n",
              cor(R$S, R$V_Q), cor(R$S, R$V_X), mean((R$S - mean(R$S))^3) / sd(R$S)^3, mean(R$ev_cut), mean(R$wt_removed)))
  invisible(R)
}
