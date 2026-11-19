## =============================================================
## nn_test_5d.R -- cross-fitted test with a neural-network witness (5-D)
## Source AFTER utils_5d.R. Three steps per batch of toys:
##   export_nn_batch_5d()   R:      toys, pushforward null fits, per-fold point sets  -> toy_XXXX.in.bin + .meta.rds
##   nn_witness_test.py     Python: trains one network per fold (GPU), writes the network's values at the
##                                  point sets and the Gauss-Newton blocks on the common measure -> toy_XXXX.out.bin
##   assemble_nn_batch_5d() R:      S_k, V_D, V_R, plug-in (push: instrument + Jacobian), v_aux, V_cross
## run_nn_study_5d() loops over batches as a single background job.
## The statistic is the one of cvgof_5d (Theory_Methods.tex two-sample extension) with the
## dictionary witness replaced by f_k = the fold's network (log intensity ratio against the
## fold's training base). Fitting randomness: Gauss-Newton linearization in the network's FREE
## parameters (|theta| < clip; parameters at the box bound respond with slope 0), damping 1
## (the "none" rule), the output bias (constant direction) damped by 1 - rho_n.
## Binary layout: see nn_witness_test.py.
## =============================================================

## base points of the pushforward null for reference rows iR at nu_s, with the
## positions (within iR) of the pre-selection events behind them
.push_base_src_5d <- function(Rpre, iR, s, cuts = CFG5$cuts) {
  P  <- .push_prep_5d(Rpre[iR, , drop = FALSE])
  f1 <- exp(s * P$r1); f2 <- exp(s * P$r2)
  kk <- pmin(P$pt1 * f1, P$pt2 * f2) > cuts$pt_min & pmax(P$pt1 * f1, P$pt2 * f2) < cuts$pt_max &
        P$mll * sqrt(f1 * f2) > cuts$mll_min & abs(P$eta1) < cuts$eta_max & abs(P$eta2) < cuts$eta_max
  list(X5 = .push_apply_5d(P, s, cuts), src = which(kk))
}

.wbin_i <- function(con, x) writeBin(as.integer(x), con, size = 4L, endian = "little")
.wbin_d <- function(con, x) writeBin(as.double(x), con, size = 8L, endian = "little")
.wbin_m <- function(con, X) writeBin(as.vector(t(X)), con, size = 4L, endian = "little")   # float32 row-major

## ---- 1. export ----
## data / N_data: draw the data from another pool sample (e.g. an EFT sample REPLACING the SM)
## with expected passing count N_data at nu = 0 (draw_toy_5d's arguments)
export_nn_batch_5d <- function(pools, dm, toy_ids, out_dir, seed = 1L, signal = NULL, NS = 0,
                               data = "SM", N_data = NULL,
                               nu_star = c(0, 0), sigma_A = CFG5$sigma_A, N0 = CFG5$N0, K = CFG5$K,
                               eps = 1e-3, cores = 8L, progress = NULL) {
  .check_dm(dm, "export_nn_batch_5d")
  out_dir <- path.expand(out_dir); dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  one <- function(b) {
    set.seed((seed * 100003 + b) %% .Machine$integer.max)                  # = run_study_5d's toy b
    t  <- add_delta_5d(draw_toy_5d(pools, N0 = N0, data = data, N_data = N_data %||% N0, signal = signal,
                                   NS = NS, nu_star = nu_star, sigma_A = sigma_A, K = K), dm)
    nf <- null_fits_push_5d(t, dm)
    nu_of <- function(k) c(nf$nu_n[nf$fold == k], nf$nu_s[nf$fold == k])
    Xref <- .push_base_src_5d(t$Rpre, seq_len(nrow(t$Rpre)), nu_of(0L)[2])$X5        # common measure
    con <- file(file.path(out_dir, sprintf("toy_%04d.in.bin", b)), "wb")
    .wbin_i(con, c(t$K, nrow(Xref))); .wbin_m(con, Xref)
    folds <- vector("list", t$K)
    for (k in seq_len(t$K)) {
      nu <- nu_of(k); iRtr <- which(t$foldRpre != k); iRte <- which(t$foldRpre == k)
      w_tr <- (t$K - 1) * N0 / (t$K * sum(t$pass0[iRtr])); w_te <- N0 / (t$K * sum(t$pass0[iRte]))
      om_tr <- w_tr * exp(nu[1]); om_te <- w_te * exp(nu[1])
      B0 <- .push_base_src_5d(t$Rpre, iRtr, nu[2])
      Bm <- .push_base_src_5d(t$Rpre, iRtr, nu[2] - eps); Bp <- .push_base_src_5d(t$Rpre, iRtr, nu[2] + eps)
      T0 <- .push_base_src_5d(t$Rpre, iRte, nu[2])
      Tp <- .push_base_src_5d(t$Rpre, iRte, nu[2] + eps); Tm <- .push_base_src_5d(t$Rpre, iRte, nu[2] - eps)
      XD <- t$xn[t$fold != k, , drop = FALSE]; iE <- which(t$fold == k); XE <- t$xn[iE, , drop = FALSE]
      ## training-base C, G (instrument d1) at nu_s and nu_s +- eps: Jacobian of the null-fit equations
      CG <- t(sapply(list(m = Bm, c = B0, p = Bp), function(B)
        c(C = om_tr * nrow(B$X5), G = om_tr * sum(.d1_5d(dm, B$X5)))))
      .wbin_i(con, c(nrow(XD), nrow(B0$X5), nrow(XE), nrow(T0$X5), nrow(Tp$X5), nrow(Tm$X5)))
      .wbin_d(con, c(om_tr, om_te))
      .wbin_m(con, XD); .wbin_m(con, B0$X5); .wbin_m(con, XE)
      .wbin_m(con, T0$X5); .wbin_m(con, Tp$X5); .wbin_m(con, Tm$X5)
      folds[[k]] <- list(nu = nu, om_tr = om_tr, om_te = om_te,
                         expo_tr = om_tr * nrow(B0$X5), expo_te = om_te * nrow(T0$X5),
                         M_te = length(iRte), te_pass0 = t$pass0[iRte], src = T0$src,
                         psiE = cbind(1, t$dD[iE, "d1"]), psiT = cbind(1, .d1_5d(dm, T0$X5)), CG = CG,
                         nE = nrow(XE), nT = nrow(T0$X5), nTp = nrow(Tp$X5), nTm = nrow(Tm$X5))
    }
    close(con)
    saveRDS(list(toy = b, K = t$K, N0 = N0, NR = t$NR, N = t$N, n_sig = t$n_sig, nu_A = t$nu_A,
                 sigma_A = t$sigma_A, nu_star = t$nu_star, nR = nrow(Xref), eps = eps, folds = folds),
            file.path(out_dir, sprintf("toy_%04d.meta.rds", b)))
    if (!is.null(progress)) cat(format(Sys.time(), "%H:%M:%S"), sprintf("exported toy %d (n_sig %d)\n", b, t$n_sig),
                                file = path.expand(progress), append = TRUE)
    b
  }
  r <- parallel::mclapply(toy_ids, function(b) try(one(b)), mc.cores = cores)
  bad <- vapply(r, inherits, logical(1), "try-error")
  if (any(bad)) stop("export_nn_batch_5d: ", sum(bad), " toys failed; first: ", r[[which(bad)[1]]])
  invisible(toy_ids)
}

## ---- read Python's per-toy output ----
.read_nn_out_5d <- function(path, meta) {
  con <- file(path, "rb"); on.exit(close(con))
  ri <- function(n) readBin(con, "integer", n, size = 4L, endian = "little")
  rf <- function(n) readBin(con, "double", n, size = 4L, endian = "little")
  rd <- function(n) readBin(con, "double", n, size = 8L, endian = "little")
  hd <- ri(2L); K <- hd[1]; P <- hd[2]
  if (K != meta$K) stop("output/meta fold mismatch in ", basename(path))
  fo <- lapply(seq_len(K), function(k) { f <- meta$folds[[k]]
    list(theta = rf(P), phiE = rd(f$nE), phiT = rd(f$nT), phiTp = rd(f$nTp), phiTm = rd(f$nTm)) })
  H  <- array(rd(K * P * P), c(P, P, K))                        # [q, p, k]; symmetric in p, q
  Mv <- rd(K * K * P * P)                                       # row-major (k, j, p, q)
  mu <- matrix(rd(K * P), K, P, byrow = TRUE)                   # (k, p)
  M  <- function(k, j) matrix(Mv[((k - 1) * K + (j - 1)) * P * P + seq_len(P * P)], P, P, byrow = TRUE)
  list(K = K, P = P, folds = fo, H = H, M = M, mean = mu)
}

.pinv_sym <- function(A, tol = 1e-10) {
  e <- eigen((A + t(A)) / 2, symmetric = TRUE); k <- e$values > tol * max(e$values, 0)
  e$vectors[, k, drop = FALSE] %*% (t(e$vectors[, k, drop = FALSE]) / e$values[k])
}

## ---- 2. assemble the statistic for one toy ----
## gn_tol: relative eigenvalue cutoff of the pseudo-inverse of H_k; gn_ridge: ridge
## (H_k + gn_ridge * mean(diag H_k) I)^-1 instead (0 = off). Both stabilize the per-toy
## V_cross by damping the near-singular free-parameter directions.
## Default gn_ridge = 1e-4: chosen on 288 null toys (nu* = 0) as the strongest stabilization that
## keeps V_cross unbiased (var(S)/mean(V0 + V_cross) = 1.02; no negative variances; size 0.059/0.003).
## clip_bias = FALSE: the biases were not clipped (--no-clip-bias), so they are always free
## free_rule = "box": parameters at the clip bound respond with slope 0 (KKT at a stationary point);
##             "all": every parameter is treated as free (bound parameters can still move with the data
##                    when the fit is early-stopped rather than stationary)
assemble_nn_toy_5d <- function(meta, out, clip, eps = meta$eps, gn_tol = 1e-10, gn_ridge = 1e-4,
                               clip_bias = TRUE, hidden = c(5, 5, 5), free_rule = c("box", "all")) {
  free_rule <- match.arg(free_rule)
  K <- meta$K; N0 <- meta$N0; P <- out$P
  sA <- meta$sigma_A; pA <- ifelse(is.finite(sA), 1 / sA^2, 0); SigAinv <- diag(pA, 2L)
  S <- S_or <- VD <- VD_raw <- VR <- VR_raw <- VX <- numeric(K); c_aux <- numeric(2L)
  sz <- c(5, hidden, 1)                                   # parameter layout W1, b1, W2, b2, ... (row-major W)
  is_bias <- unlist(lapply(seq_len(length(sz) - 1L), function(l) c(rep(FALSE, sz[l] * sz[l + 1]), rep(TRUE, sz[l + 1]))))
  if (length(is_bias) != P) stop("parameter layout does not match P = ", P)
  free <- vector("list", K); damp <- vector("list", K)
  for (k in seq_len(K)) {
    f <- meta$folds[[k]]; o <- out$folds[[k]]; om <- f$om_te
    S[k] <- sum(o$phiE) - om * sum(o$phiT)
    g <- c(om * sum(o$phiT), om * (sum(o$phiTp) - sum(o$phiTm)) / (2 * eps))
    J0 <- rbind(c(f$CG["c", "C"], (f$CG["p", "C"] - f$CG["m", "C"]) / (2 * eps)),
                c(f$CG["c", "G"], (f$CG["p", "G"] - f$CG["m", "G"]) / (2 * eps)))
    J  <- J0 + SigAinv
    Jtg <- solve(t(J), g); bk <- (K - 1) * Jtg; c_aux <- c_aux + Jtg
    ptx <- o$phiE - as.vector(f$psiE %*% bk); ptb <- o$phiT - as.vector(f$psiT %*% bk)
    S_or[k] <- sum(ptx) - om * sum(ptb)
    VD[k] <- om * sum(ptb^2); VD_raw[k] <- om * sum(o$phiT^2); VX[k] <- sum(ptx^2)
    vR <- function(vals) { v <- numeric(f$M_te); v[f$src] <- om * vals
      sum(vapply(split(v, f$te_pass0), function(x) length(x) * mean((x - mean(x))^2), numeric(1))) }
    VR[k] <- vR(ptb); VR_raw[k] <- vR(o$phiT)
    fr <- if (free_rule == "all") seq_len(P) else which(abs(o$theta) < clip * (1 - 1e-6) | (!clip_bias & is_bias))
    free[[k]] <- fr
    rho_n <- f$expo_tr / (f$expo_tr + pA[1])
    damp[[k]] <- ifelse(fr == P, 1 - rho_n, 1)                  # output bias = constant direction
  }
  v_aux <- as.numeric(crossprod(c_aux, SigAinv %*% c_aux))
  r_ratio <- N0 / meta$NR
  Hinv <- lapply(seq_len(K), function(k) { fr <- free[[k]]
    if (!length(fr)) return(matrix(0, 0, 0))
    Hk <- out$H[fr, fr, k, drop = FALSE][, , 1]; Hk <- (Hk + t(Hk)) / 2
    Hi <- if (gn_ridge > 0) solve(Hk + gn_ridge * mean(diag(Hk)) * diag(nrow(Hk))) else .pinv_sym(Hk, gn_tol)
    damp[[k]] * Hi })
  Vc <- c(data = 0, ref = 0)
  for (j in seq_len(K - 1L)) for (k in (j + 1L):K) {
    fk <- free[[k]]; fj <- free[[j]]; if (!length(fk) || !length(fj)) next
    Mkj <- out$M(k, j)[fk, fj, drop = FALSE]
    Ckj <- Mkj - tcrossprod(out$mean[k, fk], out$mean[j, fj])
    Ej <- meta$folds[[j]]$expo_te; Ek <- meta$folds[[k]]$expo_te
    for (nm in names(Vc)) { Mt <- if (nm == "data") Mkj else Mkj + r_ratio * Ckj
      Vc[nm] <- Vc[nm] + 2 * Ej * Ek * sum((Hinv[[k]] %*% Mt) * t(Hinv[[j]] %*% t(Mt))) }
  }
  St <- sum(S); Z <- function(v) St / sqrt(v)
  c(S = St, Z_fit = Z(sum(VD) + sum(VR) + v_aux + Vc[["ref"]]),
    Z_fit_D = Z(sum(VD) + sum(VR) + v_aux + Vc[["data"]]),
    Z_orth = Z(sum(VD) + sum(VR) + v_aux), Z_orthD = Z(sum(VD) + v_aux), Z_raw = Z(sum(VD_raw) + sum(VR_raw)),
    V_D = sum(VD), V_R = sum(VR), v_aux = v_aux, V_cross_data = Vc[["data"]], V_cross_ref = Vc[["ref"]],
    V_X = sum(VX), S_orth_num = sum(S_or), n_free = mean(lengths(free)),
    nu_s_hat = mean(vapply(meta$folds, function(f) f$nu[2], numeric(1))))
}

## suffix: "" for single-epoch runs, ".e<epoch>" for runs with several epoch checkpoints
assemble_nn_batch_5d <- function(toy_ids, out_dir, clip, suffix = "", cores = 8L, ...) {
  out_dir <- path.expand(out_dir)
  toy_ids <- toy_ids[file.exists(file.path(out_dir, sprintf("toy_%04d%s.out.bin", toy_ids, suffix)))]
  do.call(rbind, parallel::mclapply(toy_ids, mc.cores = cores, FUN = function(b) {
    meta <- readRDS(file.path(out_dir, sprintf("toy_%04d.meta.rds", b)))
    out  <- .read_nn_out_5d(file.path(out_dir, sprintf("toy_%04d%s.out.bin", b, suffix)), meta)
    data.frame(toy = b, method = "nn", t(assemble_nn_toy_5d(meta, out, clip, ...)), n_sig = meta$n_sig,
               smooth = NA_real_, bumps = NA_real_, secs = NA_real_)
  }))
}

## ---- 3. driver: batches of toys -> export -> python -> assemble ----
run_nn_study_5d <- function(pools, dm, B = 400L, batch = 32L, seed = 1L, signal = NULL, NS = 0,
                            data = "SM", N_data = NULL,
                            nu_star = c(0, 0), sigma_A = CFG5$sigma_A, N0 = CFG5$N0, K = CFG5$K,
                            clip = 0.72, epochs = 2800L, lr = 1e-3, hidden = "5,5,5", duty = 0.7,   # epochs may be a vector
                            clip_bias = TRUE,
                            work_dir = "nn_test", tag = "nn", python = "python3",
                            script = "nn_witness_test.py", extra_py = character(0), cores = 8L, keep_inputs = FALSE) {
  wd <- path.expand(file.path(work_dir, tag)); dir.create(wd, showWarnings = FALSE, recursive = TRUE)
  prog <- file.path(wd, "progress.log")
  say <- function(...) cat(format(Sys.time(), "%H:%M:%S"), sprintf(...), "\n", file = prog, append = TRUE)
  say("study %s: B = %d in batches of %d; data %s (N_data %g); signal %s (NS %g); nu* = (%g, %g); clip %g, epochs %s",
      tag, B, batch, data, N_data %||% N0, if (is.null(signal)) "none" else signal, NS, nu_star[1], nu_star[2], clip,
      paste(epochs, collapse = ","))
  res <- list(); ids <- split(seq_len(B), ceiling(seq_len(B) / batch))
  for (bi in seq_along(ids)) {
    tid <- ids[[bi]]; t0 <- proc.time()[["elapsed"]]
    export_nn_batch_5d(pools, dm, tid, wd, seed = seed, signal = signal, NS = NS, data = data, N_data = N_data,
                       nu_star = nu_star, sigma_A = sigma_A, N0 = N0, K = K, cores = cores)
    say("batch %d/%d: exported toys %d-%d", bi, length(ids), min(tid), max(tid))
    multi <- length(epochs) > 1L
    args <- c(script, "--dir", wd, "--toys", paste0(min(tid), "-", max(tid)), "--clip", clip, "--epochs", max(epochs),
              if (multi) c("--checkpoints", paste(epochs, collapse = ",")), "--lr", lr, "--hidden", hidden,
              "--duty", duty, "--seed", seed, if (!clip_bias) "--no-clip-bias", extra_py)
    st <- system2(python, args, stdout = file.path(wd, "python.log"), stderr = file.path(wd, "python.log"))
    if (st != 0) stop("python failed (status ", st, "); see ", file.path(wd, "python.log"))
    say("batch %d/%d: trained + evaluated", bi, length(ids))
    for (e in epochs) {
      sfx <- if (multi) sprintf(".e%d", e) else ""
      res[[as.character(e)]][[bi]] <- assemble_nn_batch_5d(tid, wd, clip, suffix = sfx, clip_bias = clip_bias)
    }
    if (!keep_inputs) unlink(file.path(wd, sprintf("toy_%04d.in.bin", tid)))
    Rl <- do.call(rbind, res[[as.character(max(epochs))]])
    say("batch %d/%d: assembled (%.1f min); running P(Z_fit > 1.645) at epoch %d = %.3f over %d toys", bi, length(ids),
        (proc.time()[["elapsed"]] - t0) / 60, max(epochs), mean(Rl$Z_fit > 1.645, na.rm = TRUE), nrow(Rl))
  }
  out <- lapply(epochs, function(e) {
    R <- do.call(rbind, res[[as.character(e)]])
    attr(R, "meta") <- list(nu_star = nu_star, sigma_A = sigma_A,
                            signal = if (data != "SM") paste0(data, " (data)") else signal, NS = NS, N0 = N0, K = K,
                            lambda_z = NA, B = B, fixed = FALSE, seed = seed, clip = clip, epochs = e)
    saveRDS(R, file.path(wd, if (length(epochs) > 1L) sprintf("results_e%d.rds", e) else "results.rds"))
    cat(sprintf("\n---- %s: clip %g, epochs %d ----\n", tag, clip, e)); summary_study_5d(R)
    R })
  names(out) <- as.character(epochs)
  say("DONE")
  invisible(if (length(epochs) == 1L) out[[1]] else out)
}

## ---- sensitivity table over (setting, clip, epoch) from saved results ----
## tags: named list setting -> list of (tag, clip); reads <work_dir>/<tag>/results_e<epoch>.rds
sens_table_5d <- function(work_dir, runs, epochs) {
  rows <- list()
  for (r in runs) for (e in epochs) {
    f <- file.path(path.expand(work_dir), r$tag, sprintf("results_e%d.rds", e)); if (!file.exists(f)) next
    R <- readRDS(f); z <- R$Z_fit[is.finite(R$Z_fit)]; V0 <- R$V_D + R$V_R + R$v_aux
    rows[[length(rows) + 1]] <- data.frame(setting = r$setting, clip = r$clip, epochs = e, toys = nrow(R),
      mean_Z = mean(z), sd_Z = sd(z), P_1.645 = mean(z > 1.645), P_2.326 = mean(z > 2.326),
      var_ratio = var(R$S) / mean(V0 + R$V_cross_ref), n_free = mean(R$n_free))
  }
  out <- do.call(rbind, rows); print(out, digits = 3, row.names = FALSE); invisible(out)
}

## ---- offline scan of the V_cross stabilization on saved null toys ----
## settings: list of named lists, e.g. list(pinv = list(gn_tol = 1e-10), r1e2 = list(gn_ridge = 1e-2))
## cores: parallel over toys. Each worker reads ONE toy's outputs and assembles it under every
## setting, returning only the summary numbers, so the main process stays small (forking a process
## that holds all toys' Gram blocks can fail). Keep OPENBLAS/OMP threads at 1.
scan_gn_5d <- function(toy_ids, out_dir, clip, settings, suffix = "", cores = 16L) {
  out_dir <- path.expand(out_dir)
  toy_ids <- toy_ids[file.exists(file.path(out_dir, sprintf("toy_%04d%s.out.bin", toy_ids, suffix)))]
  per_toy <- parallel::mclapply(toy_ids, mc.cores = cores, FUN = function(b) {
    meta <- readRDS(file.path(out_dir, sprintf("toy_%04d.meta.rds", b)))
    out  <- .read_nn_out_5d(file.path(out_dir, sprintf("toy_%04d%s.out.bin", b, suffix)), meta)
    lapply(settings, function(st) do.call(assemble_nn_toy_5d, c(list(meta = meta, out = out, clip = clip), st)))
  })
  bad <- vapply(per_toy, inherits, logical(1), "try-error"); if (any(bad)) stop(per_toy[[which(bad)[1]]])
  rows <- lapply(names(settings), function(nm) {
    R <- as.data.frame(do.call(rbind, lapply(per_toy, `[[`, nm)))
    V0 <- R$V_D + R$V_R + R$v_aux; Vc <- R$V_cross_ref; ok <- is.finite(R$Z_fit)
    data.frame(setting = nm, toys = nrow(R), ratio_all = var(R$S) / mean(V0 + Vc),
               need_over_V0 = (var(R$S) - mean(V0)) / mean(V0), Vc_over_V0 = mean(Vc) / mean(V0),
               n_neg = sum(V0 + Vc <= 0), cor_Vc_V0 = cor(Vc, V0),
               q05 = quantile(Vc / V0, .05), q50 = quantile(Vc / V0, .5), q95 = quantile(Vc / V0, .95),
               sd_Z = sd(R$Z_fit[ok]), size05 = mean(R$Z_fit[ok] > 1.645), size01 = mean(R$Z_fit[ok] > 2.326),
               row.names = NULL)
  })
  out <- do.call(rbind, rows); print(out, digits = 3, row.names = FALSE); invisible(out)
}
