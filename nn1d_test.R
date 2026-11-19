## =============================================================
## nn1d_test.R -- the 1-D cross-fitted test (Poisson mode, utils.R) with a neural-network witness.
## Requires source("utils.R") first. Per batch of toys:
##   export_nn1d_toy()    R:      toy, folds, per-fold constrained null fits and quadrature grids
##                                -> toy_XXXX.in.bin + toy_XXXX.meta.rds
##   nn1d_test.py         Python: one network per fold (GPU), its values at the held-out events and
##                                on the fold's grid, Gauss-Newton blocks on the common reference
##                                measure (fold 1's grid) -> toy_XXXX.out.bin
##   assemble_nn1d_toy()  R:      fold records -> cvgof_test_quad() (S, V_Q, plug-in, v_aux), plus
##                                V_cross from the network's FREE parameters
## The witness phi_k = f_k is the fold's network (log intensity ratio against the fold's fitted null),
## exactly where the dictionary's fit_tilt(normalize = FALSE) puts phi. Fitting randomness, the 5-D
## rule unchanged: Gauss-Newton linearization in the free parameters (parameters at the clip bound
## respond with slope 0), damping 1 ("none"), the output bias (constant direction) damped by
## 1 - rho_n, ridge (H + r * mean diag H * I)^-1 with r = 1e-4. No reference sample in 1-D, so no
## V_R and no r C term.
##   V_cross = sum_{j != k} E_j E_k tr(D_k H_k^-1 M_kj D_j H_j^-1 M_jk)
## Headline: Z_fit = S / sqrt(V_Q(orth) + v_aux + V_cross).
## =============================================================

.wi <- function(con, x) writeBin(as.integer(x), con, size = 4L, endian = "little")
.wf <- function(con, x) writeBin(as.double(x), con, size = 4L, endian = "little")
.wd <- function(con, x) writeBin(as.double(x), con, size = 8L, endian = "little")

.nn1d_frames <- function(x, folds, nu_A, sigma_A, N0, K, n_grid) {
  null <- null_exponential()
  lapply(seq_len(K), function(k) {
    fr <- fold_null_frame("pois", x[folds != k], N_k = sum(folds == k), K = K, null = null,
                          nu_A = nu_A, sigma_A = sigma_A, N0 = N0)
    list(fr = fr, g = make_quad_grid(null, fr$theta, n_grid = n_grid))
  })
}

export_nn1d_toy <- function(b, dir, seed, sig, signal, nu_star, sigma_A, N0, K, n_grid) {
  set.seed(seed * 100003 + b)
  d <- gen_data_pois(sig = sig, nu_star = nu_star, sigma_A = sigma_A, N0 = N0, signal = signal)
  folds <- make_folds(d$N, K, balanced = TRUE)
  fg <- .nn1d_frames(d$x, folds, d$nu_A, sigma_A, N0, K, n_grid)
  con <- file(file.path(dir, sprintf("toy_%04d.in.bin", b)), "wb"); on.exit(close(con))
  .wi(con, c(K, n_grid)); .wf(con, fg[[1]]$g$x_grid); .wd(con, fg[[1]]$g$wp)
  for (k in seq_len(K)) {
    fr <- fg[[k]]$fr; g <- fg[[k]]$g
    x_tr <- d$x[folds != k]; x_ev <- d$x[folds == k]
    .wi(con, c(length(x_tr), length(x_ev), n_grid)); .wd(con, fr$expo_tr)
    .wf(con, x_tr); .wf(con, x_ev); .wf(con, g$x_grid); .wd(con, fr$expo_tr * g$wp)
  }
  saveRDS(list(x = d$x, folds = folds, nu_A = d$nu_A, n_sig = d$n_sig, N = d$N),
          file.path(dir, sprintf("toy_%04d.meta.rds", b)))
  invisible(b)
}

.read_nn1d_out <- function(path) {
  con <- file(path, "rb"); on.exit(close(con))
  ri <- function(n) readBin(con, "integer", n, size = 4L, endian = "little")
  rd <- function(n) readBin(con, "double", n, size = 8L, endian = "little")
  KP <- ri(2); K <- KP[1]; P <- KP[2]
  folds <- lapply(seq_len(K), function(k) {
    n <- ri(2)
    list(phiE = rd(n[1]), phiG = rd(n[2]), free = ri(P) == 1L, theta = rd(P),
         H = t(matrix(rd(P * P), P, P)))
  })
  A <- array(rd(K * K * P * P), dim = c(P, P, K, K))          # C order (k, j, p, q) -> A[q, p, j, k]
  list(K = K, P = P, folds = folds, M = function(k, j) t(A[, , j, k]))
}

## V_cross from Gauss-Newton blocks; ridge = NULL -> plain inverse (NA if singular)
.vcross_nn1d <- function(o, expo_te, rho_n, ridge) {
  K <- o$K; P <- o$P
  Hinv <- lapply(seq_len(K), function(k) {
    fr <- o$folds[[k]]$free; H <- o$folds[[k]]$H[fr, fr, drop = FALSE]
    if (!is.null(ridge)) H <- H + ridge * mean(diag(H)) * diag(nrow(H))
    Hi <- tryCatch(solve(H), error = function(e) NULL); if (is.null(Hi)) return(NULL)
    damp <- ifelse(which(fr) == P, 1 - rho_n[k], 1)
    damp * Hi                                                  # D_k H_k^-1 (row scaling)
  })
  if (any(vapply(Hinv, is.null, logical(1)))) return(NA_real_)
  V <- 0
  for (j in seq_len(K - 1L)) for (k in (j + 1L):K) {
    Mkj <- o$M(k, j)[o$folds[[k]]$free, o$folds[[j]]$free, drop = FALSE]
    V <- V + 2 * expo_te[j] * expo_te[k] * sum((Hinv[[k]] %*% Mkj) * t(Hinv[[j]] %*% t(Mkj)))
  }
  V
}

assemble_nn1d_toy <- function(b, dir, sigma_A, N0, K, n_grid, ridge = 1e-4, suffix = "") {
  m  <- readRDS(file.path(dir, sprintf("toy_%04d.meta.rds", b)))
  o  <- .read_nn1d_out(file.path(dir, sprintf("toy_%04d%s.out.bin", b, suffix)))
  fg <- .nn1d_frames(m$x, m$folds, m$nu_A, sigma_A, N0, K, n_grid)
  pf <- lapply(seq_len(K), function(k) {
    idx <- which(m$folds == k)
    stopifnot(length(o$folds[[k]]$phiE) == length(idx), length(o$folds[[k]]$phiG) == n_grid)
    make_fold_record(fg[[k]]$fr, fg[[k]]$g$x_grid, fg[[k]]$g$wp, idx, m$x[idx],
                     o$folds[[k]]$phiE, o$folds[[k]]$phiG)
  })
  r <- cvgof_test_quad(list(per_fold = pf, n = m$N, mode = "pois"))
  expo_te <- vapply(fg, function(z) z$fr$expo_te, 0); rho_n <- vapply(fg, function(z) z$fr$shrink[["nu_n"]], 0)
  Vc  <- .vcross_nn1d(o, expo_te, rho_n, ridge)
  Vc0 <- .vcross_nn1d(o, expo_te, rho_n, NULL)
  S <- r$orth$S; VQ <- r$orth$V_q; VX <- r$orth$V_x; va <- r$v_aux
  data.frame(toy = b, n_sig = m$n_sig, N = m$N, S = S, V_Q = VQ, V_X = VX, v_aux = va,
             V_cross = Vc, V_cross0 = Vc0,
             n_free = mean(vapply(o$folds, function(f) sum(f$free), 0)),
             Z_fit = S / sqrt(VQ + va + Vc), Z_fit_X = S / sqrt(VX + va + Vc),
             Z_fit0 = S / sqrt(VQ + va + Vc0), Z_orth = r$orth$Z_Q, Z_raw = r$raw$Z_Q)
}

## ---- one study: batches of export -> python -> assemble, as a single background job ----
## epochs: one value, or several (one training run, outputs at each; results_e<epoch>.rds per epoch).
## kappa: NULL = the raw network witness; a number = null-tail taper (needs source("nn1d_trunc.R")).
run_nn1d_study <- function(B = 400L, seed = 1L, sig = "NP1", signal = FALSE, nu_star = c(0, 0),
                           sigma_A = c(0.05, 0.05), N0 = N0_DEFAULT, K = K_FOLDS, n_grid = 4001L,
                           clip = 4, bias_clip = FALSE, epochs = 30000L, lr = 1e-3, hidden = 4L, kappa = NULL,
                           duty = 1, batch = 100L, ridge = 1e-4, tag = NULL,
                           work_dir = "nn1d_test", script = "nn1d_test.py", python = "python3",
                           extra_py = character(), cores = 8L, keep_inputs = FALSE, keep_outputs = TRUE) {
  epochs <- sort(as.integer(epochs)); multi <- length(epochs) > 1L
  if (!is.null(kappa) && !exists("assemble_nn1d_trunc")) stop("kappa needs source(\"nn1d_trunc.R\")")
  if (is.null(tag)) tag <- sprintf("%s_%s_c%g%s_e%s%s_s%d", if (signal) sig else "null",
                                   if (all(nu_star == 0)) "nu0" else sprintf("nu%g_%g", nu_star[1], nu_star[2]),
                                   clip, if (bias_clip) "" else "w",
                                   if (multi) paste0(epochs %/% 1000L, "k", collapse = "-") else as.character(epochs),
                                   if (is.null(kappa)) "" else sprintf("_tap%g", kappa), seed)
  wd <- path.expand(file.path(work_dir, tag)); dir.create(wd, showWarnings = FALSE, recursive = TRUE)
  prog <- file.path(wd, "progress.log")
  say <- function(...) { s <- sprintf(...); cat(format(Sys.time(), "%H:%M:%S"), s, "\n", file = prog, append = TRUE)
                         cat(format(Sys.time(), "%H:%M:%S"), s, "\n") }
  say("study %s: B %d, seed %d, %s, nu* (%g, %g), sigma_A (%g, %g); clip %g%s, epochs %s%s", tag, B, seed,
      if (signal) sig else "null", nu_star[1], nu_star[2], sigma_A[1], sigma_A[2], clip,
      if (bias_clip) "" else " (weights only)", paste(epochs, collapse = ","),
      if (is.null(kappa)) "" else sprintf(", taper kappa %g", kappa))
  sfx_of <- function(e) if (multi) sprintf(".e%d", e) else ""
  asm <- function(b, e) {
    if (is.null(kappa)) assemble_nn1d_toy(b, wd, sigma_A, N0, K, n_grid, ridge, suffix = sfx_of(e))
    else assemble_nn1d_trunc(b, wd, 0L, sigma_A, N0, K, n_grid, ridge, kappa = kappa, suffix = sfx_of(e))
  }
  ids <- split(seq_len(B), ceiling(seq_len(B) / batch)); res <- list(); t0 <- proc.time()[["elapsed"]]
  for (bi in seq_along(ids)) {
    tid <- ids[[bi]]
    ok <- parallel::mclapply(tid, function(b) try(export_nn1d_toy(b, wd, seed, sig, signal, nu_star, sigma_A, N0, K, n_grid)),
                             mc.cores = cores)
    if (any(vapply(ok, inherits, logical(1), "try-error"))) stop("export failed in batch ", bi)
    args <- c(script, "--dir", wd, "--toys", paste0(min(tid), "-", max(tid)), "--clip", clip,
              if (!bias_clip) "--no-bias-clip", "--epochs", max(epochs),
              if (multi) c("--checkpoints", paste(epochs, collapse = ",")), "--lr", lr, "--hidden", hidden,
              "--duty", duty, "--seed", seed, extra_py)
    st <- system2(python, args)
    if (st != 0) stop("nn1d_test.py failed in batch ", bi, " (status ", st, ")")
    for (e in epochs) {
      rr <- parallel::mclapply(tid, function(b) try(asm(b, e)), mc.cores = cores)
      bad <- vapply(rr, inherits, logical(1), "try-error"); if (any(bad)) stop("assembly failed: ", rr[[which(bad)[1]]])
      res[[as.character(e)]][[bi]] <- do.call(rbind, rr)
      if (!keep_outputs) unlink(file.path(wd, sprintf("toy_%04d%s.out.bin", tid, sfx_of(e))))
    }
    if (!keep_inputs) unlink(file.path(wd, sprintf("toy_%04d.in.bin", tid)))
    R <- do.call(rbind, res[[as.character(max(epochs))]])
    say("batch %d/%d done (%.1f min): %d toys; at epoch %d: running var(S)/mean(V_Q+v_aux+V_cross) %.3f, P(Z_fit > 1.645) %.3f",
        bi, length(ids), (proc.time()[["elapsed"]] - t0) / 60, nrow(R), max(epochs),
        var(R$S) / mean(R$V_Q + R$v_aux + R$V_cross), mean(R$Z_fit > 1.645))
  }
  out <- lapply(epochs, function(e) {
    R <- do.call(rbind, res[[as.character(e)]])
    attr(R, "meta") <- list(tag = tag, B = B, seed = seed, sig = if (signal) sig else "none", nu_star = nu_star,
                            sigma_A = sigma_A, N0 = N0, K = K, clip = clip, bias_clip = bias_clip,
                            epochs = e, lr = lr, hidden = hidden, ridge = ridge, kappa = kappa,
                            mock = "--mock" %in% extra_py)
    saveRDS(R, file.path(wd, if (multi) sprintf("results_e%d.rds", e) else "results.rds"))
    summary_nn1d(R); R })
  names(out) <- as.character(epochs)
  say("DONE")
  invisible(if (multi) out else out[[1]])
}

## ---- sensitivity table over (clip, epochs): null size and power, nominal and size-matched ----
## runs: list of list(setting = "null" | "NP1" | ..., clip, dir = study directory); epochs: the checkpoints
sens_table_nn1d <- function(runs, epochs) {
  rd <- function(d, e) { f <- file.path(path.expand(d), sprintf("results_e%d.rds", e)); if (file.exists(f)) readRDS(f) else NULL }
  rows <- list()
  for (cl in unique(vapply(runs, `[[`, 0, "clip"))) for (e in epochs) {
    rs <- Filter(function(r) r$clip == cl, runs); nl <- Filter(function(r) r$setting == "null", rs)
    R0 <- if (length(nl)) rd(nl[[1]]$dir, e) else NULL; if (is.null(R0)) next
    q95 <- quantile(R0$Z_fit, .95)
    row <- data.frame(clip = cl, epochs = e, ratio = var(R0$S) / mean(R0$V_Q + R0$v_aux + R0$V_cross),
                      Z_mean = mean(R0$Z_fit), sd_Z = sd(R0$Z_fit), size05 = mean(R0$Z_fit > qnorm(.95)),
                      size01 = mean(R0$Z_fit > qnorm(.99)), q95 = unname(q95), n_free = mean(R0$n_free))
    for (r in Filter(function(r) r$setting != "null", rs)) {
      R1 <- rd(r$dir, e); if (is.null(R1)) next
      row[[paste0(r$setting, "_med")]] <- median(R1$Z_fit)
      row[[paste0(r$setting, "_pow")]] <- mean(R1$Z_fit > qnorm(.95))
      row[[paste0(r$setting, "_pow_sm")]] <- mean(R1$Z_fit > q95)
    }
    rows[[length(rows) + 1]] <- row
  }
  out <- do.call(rbind, rows); print(format(out, digits = 3), row.names = FALSE); invisible(out)
}

summary_nn1d <- function(R) {
  m <- attr(R, "meta")
  cat(sprintf("study %s: %d toys, signal %s, nu* = (%g, %g), sigma_A = (%g, %g); clip %g%s, epochs %d%s, ridge %g%s\n",
              m$tag, nrow(R), m$sig, m$nu_star[1], m$nu_star[2], m$sigma_A[1], m$sigma_A[2], m$clip,
              if (isTRUE(m$bias_clip)) "" else " (weights only)", m$epochs,
              if (is.null(m$kappa)) "" else sprintf(", taper kappa %g", m$kappa), m$ridge, if (isTRUE(m$mock)) ", MOCK" else ""))
  base <- R$V_Q + R$v_aux
  cat(sprintf("  var(S)/mean(V_Q+v_aux) = %.3f;  var(S)/mean(... + V_cross) = %.3f  [no ridge: %.3f, %d NA];  mean S = %.2f (se %.2f)\n",
              var(R$S) / mean(base), var(R$S) / mean(base + R$V_cross),
              var(R$S[is.finite(R$V_cross0)]) / mean((base + R$V_cross0)[is.finite(R$V_cross0)]), sum(!is.finite(R$V_cross0)),
              mean(R$S), sd(R$S) / sqrt(nrow(R))))
  cat(sprintf("  V_cross/V_Q mean %.3f;  free parameters per network %.1f;  negative totals %d\n",
              mean(R$V_cross / R$V_Q), mean(R$n_free), sum(base + R$V_cross <= 0)))
  zc <- c("Z_fit", "Z_fit_X", "Z_fit0", "Z_orth", "Z_raw"); zc <- zc[vapply(zc, function(z) any(is.finite(R[[z]])), TRUE)]
  tab <- t(vapply(zc, function(z) { v <- R[[z]][is.finite(R[[z]])]
    c(mean = mean(v), sd = sd(v), median = median(v), `P>1.645` = mean(v > qnorm(.95)), `P>2.326` = mean(v > qnorm(.99))) },
    numeric(5)))
  print(round(tab, 3))
  invisible(tab)
}
