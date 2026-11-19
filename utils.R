## =============================================================
## Prototypical example: Exp(theta) null vs sparse-mixture alt
## Modules 1/3 scaffolding: data, folds, CV witness
## =============================================================

RATE0   <- 1         # Exp(1) bulk component
K_FOLDS <- 10
if (!exists("SIG")) SIG <- "NP1"

## ---------------------------------------------------------------
## Signal registry (NPLM 1-D benchmarks)
## ---------------------------------------------------------------
## NPLM writes n(x|NPi;nu) = n(x|R_nu) + N_i f_i(x) on top of
## N(R0) = 2000 expected reference events. Normalising:
##   lambda = N_i / (2000 + N_i),  n_def = 2000 + N_i
## lambda is a property of the model; n is a free argument, so power
## studies can scale n at fixed lambda. All three n_def are divisible
## by K = 10.
SIGNALS <- list(
  NP1 = list(N_i =  10, lambda =  10/2010, n_def = 2010,
             r     = function(m) rnorm(m, 6.4, 0.16),
             d     = function(x) dnorm(x, 6.4, 0.16),
             label = "N(6.4, 0.16^2) sparse spike"),
  NP2 = list(N_i = 180, lambda = 180/2180, n_def = 2180,
             r     = function(m) rgamma(m, shape = 3, rate = 1),
             d     = function(x) dgamma(x, shape = 3, rate = 1),
             label = "Gamma(3,1) = (x^2/2) exp(-x)"),
  NP3 = list(N_i =  90, lambda =  90/2090, n_def = 2090,
             r     = function(m) rnorm(m, 1.6, 0.16),
             d     = function(x) dnorm(x, 1.6, 0.16),
             label = "N(1.6, 0.16^2) bulk peak")
)

## ---------------------------------------------------------------
## Data generation
## ---------------------------------------------------------------
## n = NULL      -> SIGNALS[[sig]]$n_def (2010 / 2180 / 2090); pass n
##                  explicitly to scale at fixed lambda
## n_sig = NULL  -> Binomial(n, lambda) contamination count (the
##                  model as written)
## n_sig = 10    -> exactly 10 signal points, for variance-reduced
##                  power studies at fixed N_Sig
## sig           -> name in SIGNALS; the component's location/scale
##                  are properties of the model, not arguments
## lambda = NULL -> SIGNALS[[sig]]$lambda; override for power-vs-
##                  lambda curves at fixed component shape
gen_data <- function(n = NULL, sig = SIG, lambda = NULL,
                     rate = RATE0, signal = TRUE, n_sig = NULL) {
  sg <- SIGNALS[[sig]]
  if (is.null(sg)) stop("gen_data: unknown sig '", sig, "'")
  if (is.null(n))      n      <- sg$n_def
  if (is.null(lambda)) lambda <- sg$lambda
  if (!signal) {
    n1 <- 0L
  } else if (is.null(n_sig)) {
    n1 <- rbinom(1L, n, lambda)
  } else {
    n1 <- as.integer(n_sig)
  }
  x <- c(rexp(n - n1, rate = rate), sg$r(n1))
  z <- c(rep(0L, n - n1), rep(1L, n1))
  ord <- sample(n)                      # de-block the labels
  list(x = x[ord], z = z[ord], n_sig = n1)
}
## RNG stream is unchanged from the pre-registry version: rbinom ->
## rexp(n - n1) -> component draw of length n1 -> sample(n). For
## sig = "NP1" the output is bit-identical under the same seed.
##
## Neither Normal component is truncated at 0. NP1 sits 40 sd from
## the boundary; NP3 sits 10 sd out, P(X < 0) ~ 8e-24, so the
## untruncated draw still matches the density as written in
## Theory_Methods.md to machine precision. NP2's Gamma(3,1) is
## supported on (0, Inf) natively.

## ---------------------------------------------------------------
## KDE with CV bandwidth
## ---------------------------------------------------------------
## Gaussian kernel with reflection at 0. The reflection is not
## cosmetic here: ~2000 of 2010 points sit in [0, 3], and an
## uncorrected KDE loses mass across the boundary, biasing
## log(p_alt/p_theta) systematically over the bulk. That bias has a
## score-direction component, i.e. it inflates g_k, which is exactly
## what Module 3's orthogonalization then has to remove.
dkde_refl <- function(xnew, xtrain, h) {
  D <- outer(xnew, xtrain, "-") / h
  S <- outer(xnew, xtrain, "+") / h
  rowMeans(dnorm(D) + dnorm(S)) / h
}

## Bandwidth grid tied to the scale of the training data:
## from (range / n_tr), roughly the mean interpoint spacing, up to
## (range / 10), which is fully oversmoothed. The grid resolution
## grows with n: ceiling(2 * log2(n)) points spans the same
## [2r/n, r/10] range with adjacent-h ratio (n/20)^(1/(n_grid-1)),
## so the quantization of the selected h tightens slowly as n grows
## while bw_lcv() stays O(n^2 * n_grid).
make_h_grid <- function(xtr, n_grid = ceiling(2 * log2(length(xtr)))) {
  r <- diff(range(xtr))
  n <- length(xtr)
  stopifnot(n_grid >= 2L, r > 0)
  exp(seq(log(2 * r / n), log(r / 10), length.out = n_grid))
}

## Leave-one-out likelihood CV. LCV (not LSCV/bw.ucv) is the right
## criterion here because the witness is a log-density ratio, so the
## loss the bandwidth should minimize is KL, not ISE.
bw_lcv <- function(xtr, h_grid, floor_dens = 1e-12) {
  n <- length(xtr)
  D <- outer(xtr, xtr, "-")
  S <- outer(xtr, xtr, "+")
  dg <- cbind(seq_len(n), seq_len(n))
  cv <- vapply(h_grid, function(h) {
    W <- dnorm(D / h) + dnorm(S / h)
    W[dg] <- 0                       # drops own kernel AND own reflection
    f_loo <- rowSums(W) / ((n - 1) * h)
    mean(log(pmax(f_loo, floor_dens)))
  }, numeric(1))
  list(h = h_grid[which.max(cv)], cv = cv, grid = h_grid)
}


## =============================================================
## Module 3: cross-fitted witness construction
## =============================================================

cvgof_fit_witness <- function(x, K = 10, n_y_mult = 20, eps_mix = 1e-3,
                              h = NULL, seed = NULL) {

  if (!is.null(seed)) set.seed(seed)
  n <- length(x)
  stopifnot(n %% K == 0, eps_mix > 0, eps_mix < 1)

  ## --- fold assignment: independent of the data, per (B1) ---------
  fold_id <- sample(rep(seq_len(K), each = n / K))

  ## --- bandwidth --------------------------------------------------
  ## DEVIATION: selecting h on the full sample makes h depend on every
  ## fold's evaluation data, so phi_k is not strictly D_{-k}-measurable.
  ## Accepted deliberately to remove fold-to-fold bandwidth noise, which
  ## works against the stabilization in (B5). Pass h explicitly to use a
  ## fold-honest or auxiliary-sample bandwidth instead.
  lcv <- NULL
  if (is.null(h)) {
    h_grid <- make_h_grid(x)
    lcv    <- bw_lcv(x, h_grid)
    h      <- lcv$h
    i_star <- which.max(lcv$cv)
    if (i_star == 1L || i_star == length(h_grid))
      warning("LCV argmax at grid endpoint ", i_star, " (h = ",
              signif(h, 3), "): widen make_h_grid()")
  }

  ## --- per-fold loop ----------------------------------------------
  per_fold <- vector("list", K)

  for (k in seq_len(K)) {
    idx_ev <- which(fold_id == k)
    xtr    <- x[fold_id != k]
    x_ev   <- x[idx_ev]

    ## null MLE (rate) from the training folds only
    theta_k <- 1 / mean(xtr)

    ## Calibration draws come from the FOLD-SPECIFIC null MLE. The
    ## calibration target is K^{-1} sum_k m_k(theta^{(-k)}), fold by
    ## fold; a single global theta-hat would both target the wrong
    ## quantity and depend on D_k.
    n_y <- n_y_mult * length(x_ev)
    y   <- rexp(n_y, rate = theta_k)

    ## KDE fitted on D_{-k}, evaluated on both sides with the same h
    f_kde_x <- dkde_refl(x_ev, xtr, h)
    f_kde_y <- dkde_refl(y,    xtr, h)

    f_nul_x <- dexp(x_ev, rate = theta_k)
    f_nul_y <- dexp(y,    rate = theta_k)

    ## Shrink the alternative toward the fitted null: bounds phi below
    ## by log(eps_mix), stays D_{-k}-measurable, and is invisible where
    ## the KDE has data. Replaces a pmax() floor, which would leave
    ## phi ~ -19 at tail points and dominate Var(phi).
    f_alt_x <- (1 - eps_mix) * f_kde_x + eps_mix * f_nul_x
    f_alt_y <- (1 - eps_mix) * f_kde_y + eps_mix * f_nul_y

    phi_x <- log(f_alt_x) - log(f_nul_x)
    phi_y <- log(f_alt_y) - log(f_nul_y)

    per_fold[[k]] <- list(
      x_ev   = x_ev,
      phi_x  = phi_x,
      y      = y,          # kept: g_k needs cov(phi_y, y), not phi_y alone
      phi_y  = phi_y,
      theta  = theta_k,    # scalar rate; the null density needs nothing more
      idx_ev = idx_ev,
      n_clip = c(x = sum(f_kde_x < eps_mix * f_nul_x),
                 y = sum(f_kde_y < eps_mix * f_nul_y))
    )
  }

  list(per_fold = per_fold, fold_id = fold_id, h = h,
       lcv = lcv, eps_mix = eps_mix, call = match.call())
}


## --- reassembly helper: length-n Phi in original order ------------
cvgof_phi_vector <- function(out) {
  phi <- numeric(length(out$fold_id))
  for (fk in out$per_fold) phi[fk$idx_ev] <- fk$phi_x
  phi
}

## =============================================================
## Module 3: orthogonalized test statistic
## =============================================================
## Exponential null, rate parameterization:
##   s_theta(x) = 1/theta - x           (score)
##   I(theta)   = 1/theta^2             (information)
##   psi(x)     = I^{-1} s = theta - theta^2 x
##   g_k        = Cov_{p_theta}(phi_k, s) = -Cov_{p_theta}(phi_k, X)
##   phi~_k     = phi_k - g_k * psi_k                       (3.8)
##
## Because E_{p_theta}[psi] = 0 exactly, orthogonalization does not
## move the calibration target: m~_k(theta) = m_k(theta). It only
## removes the score-direction variance. So phi~ is applied to BOTH
## the held-out data and the calibration draws.
##
## Two axes, giving a 2x2 of studentizations:
##   ROW    raw (option 1) vs orthogonalized (option 2). The row fixes
##          the NUMERATOR too -- T^orth differs from T by
##          K^{-1} sum_k g_k * mean(psi(x_ev_k)), an O_p(n^{-1/2})
##          shift -- so a raw numerator cannot take an orthogonalized
##          variance. Not a free knob.
##   COLUMN evaluation variance from the held-out X, or from the
##          calibration draws Y. This IS free. Under H0 both converge
##          to the same limit by (A5)/(A7); under H1 the X-side
##          absorbs the signal points and inflates the SE.
## Both columns are always computed and returned; the headline Z is
## the orthogonalized row with the Y-side variance. Z_min (the min-
## variance rule) is returned for diagnostics but is not the default,
## since in finite samples it selects the X column almost always.

cvgof_test <- function(x, cv_out, alpha = 0.05) {

  K <- length(cv_out$per_fold)
  n <- length(cv_out$fold_id)
  stopifnot(length(x) == n)

  ## alignment check: cheap, and catches a reordered x immediately
  for (k in seq_len(K)) {
    fk <- cv_out$per_fold[[k]]
    stopifnot(isTRUE(all.equal(fk$x_ev, x[fk$idx_ev])),
              length(fk$phi_x) == length(fk$x_ev),
              length(fk$phi_y) == length(fk$y))
  }

  g_hat    <- numeric(K)
  Ty_raw   <- numeric(K)   # per-fold calibration means, raw witness
  Ty_orth  <- numeric(K)   #                             orthogonalized
  Vy_raw   <- numeric(K)   # UNnormalized per-fold calibration variances
  Vy_orth  <- numeric(K)
  ny_vec   <- numeric(K)
  phi_all  <- numeric(n)   # raw witness,   original order
  phit_all <- numeric(n)   # orthogonalized, original order

  for (k in seq_len(K)) {
    fk  <- cv_out$per_fold[[k]]
    th  <- fk$theta
    ny_vec[k] <- length(fk$y)

    psi_x <- th - th^2 * fk$x_ev
    psi_y <- th - th^2 * fk$y

    ## g_k under p_{theta^{(-k)}}, from the calibration draws. It is a
    ## null-model object: using x here would estimate Cov_p, which
    ## under H1 picks up the signal and would orthogonalize away the
    ## very thing the test is looking for.
    g_hat[k] <- -stats::cov(fk$phi_y, fk$y)

    phit_x <- fk$phi_x - g_hat[k] * psi_x
    phit_y <- fk$phi_y - g_hat[k] * psi_y

    phi_all[fk$idx_ev]  <- fk$phi_x
    phit_all[fk$idx_ev] <- phit_x

    Ty_raw[k]  <- mean(fk$phi_y)
    Ty_orth[k] <- mean(phit_y)
    Vy_raw[k]  <- stats::var(fk$phi_y)   # evaluation variance, null side
    Vy_orth[k] <- stats::var(phit_y)
  }

  ## The 2x2. Under H0 all four share the limit V_* (option-2 cells
  ## share the effective information zeta_*^2 = V - g^2 I^{-1}), so
  ## the null exhibit is that the whole table agrees to MC error.
  ## Down a column: Remark 3's sandwich ratio (witness orthogonality).
  ## Across a row: X/Y. Asymptotically 1 under H0 and >1 under H1, but
  ## in finite samples the Y draws (n_y_mult x more numerous) reach
  ## further into the tail than the held-out X, where phi -> log(eps_mix);
  ## V_Y therefore runs well above V_X under H0. See EXPERIMENTS.md.
  V_tab <- rbind(X = c(raw  = stats::var(phi_all),
                       orth = stats::var(phit_all)),
                 Y = c(raw  = mean(Vy_raw),
                       orth = mean(Vy_orth)))

  ## Simulation-noise term: Var of T_Y = K^{-2} sum_k V_k / n_y,k.
  ## NOTE in the Y column both terms trace to the same Vy, so the
  ## total is Vy * (1/n + 1/(K*n_y)) and the split is fixed by
  ## n_y_mult rather than estimated -- the two-term decomposition
  ## stops being an informative diagnostic there.
  vsim_raw  <- sum(Vy_raw  / ny_vec) / K^2
  vsim_orth <- sum(Vy_orth / ny_vec) / K^2

  stat <- function(Tx, Ty, V_x, V_y, vsim) {
    TY  <- mean(Ty)
    tau <- Tx - TY
    z_of <- function(V) tau / sqrt(V / n + vsim)
    list(T_x = Tx, T_Y = TY, tau = tau,
         Z_X = z_of(V_x), Z_Y = z_of(V_y), Z_min = z_of(min(V_x, V_y)),
         se_X = sqrt(V_x / n + vsim), se_Y = sqrt(V_y / n + vsim),
         V_x = V_x, V_y = V_y, v_sim = vsim)
  }

  orth <- stat(mean(phit_all), Ty_orth, V_tab["X", "orth"],
               V_tab["Y", "orth"], vsim_orth)
  raw  <- stat(mean(phi_all),  Ty_raw,  V_tab["X", "raw"],
               V_tab["Y", "raw"],  vsim_raw)

  structure(list(
    Z          = orth$Z_Y,      # headline: orthogonalized, null-side variance
    p_value    = stats::pnorm(orth$Z_Y, lower.tail = FALSE),
    reject     = stats::pnorm(orth$Z_Y, lower.tail = FALSE) < alpha,
    orth       = orth,
    raw        = raw,
    g_hat      = g_hat,
    V_tab      = V_tab,
    sandwich_X = V_tab["X", "raw"] / V_tab["X", "orth"],
    sandwich_Y = V_tab["Y", "raw"] / V_tab["Y", "orth"],
    ratio_raw  = V_tab["X", "raw"]  / V_tab["Y", "raw"],
    ratio_orth = V_tab["X", "orth"] / V_tab["Y", "orth"],
    phi        = phi_all,
    phi_orth   = phit_all,
    h          = cv_out$h,
    n          = n, K = K, alpha = alpha
  ), class = "cvgof_test")
}

## =============================================================
## Toy loop: null or alternative experiments
## =============================================================
## Assumes utils.R has already been sourced (these live in it).
## Usage:
##   source("utils.R")
##   np1 <- run_cvgof_experiment(signal = TRUE, n = 2010,
##                               h_fix = 0.25, eps_mix = 0.02)
##
## Auto tag encodes (signal, sig, n, h_fix, eps_mix, n_y_mult, seed);
## n_toys is fixed at 400 by convention and deliberately left out.
## Decimal points become "p" so the names stay glob- and
## extension-safe. sig appears only on ALT tags: the null DGP is
## Exp(rate) regardless of sig, so a null run is shared across
## scenarios at matching (n, h_fix, eps_mix, n_y_mult, K) and must
## not be named as if it belonged to one.
##   alt_NP2_n2180_h0p25_eps0p02_ny20_s1001
##   null_n4020_hLCV_eps0p02_ny5_s1002

.fmt_num <- function(v) sub("\\.", "p", format(v, trim = TRUE))

run_cvgof_experiment <- function(signal   = TRUE,
                                 n        = NULL,
                                 h_fix    = 0.25,
                                 eps_mix  = 0.02,
                                 n_y_mult = 20,
                                 n_toys   = 400,
                                 K        = K_FOLDS,
                                 seed     = 1001,
                                 cores    = 1,
                                 chunk    = 25,
                                 tag      = NULL,
                                 save_dir = ".",
                                 verbose  = TRUE,
                                 sig      = SIG,
                                 overwrite = FALSE) {

  ## validate sig before anything forks: SIGNALS[[typo]] is NULL and
  ## would otherwise fail inside a worker, minutes in.
  stopifnot(is.character(sig), length(sig) == 1L,
            sig %in% names(SIGNALS))
  sg <- SIGNALS[[sig]]
  if (is.null(n)) {
    if (!signal) stop("run_cvgof_experiment: n must be given explicitly ",
                      "when signal = FALSE (the null DGP does not depend ",
                      "on sig; n must match the ALT run being calibrated)")
    n <- sg$n_def
  }

  stopifnot(n %% K == 0, n_toys >= 1, chunk >= 1,
            eps_mix > 0, eps_mix < 1, n_y_mult >= 1)

  if (is.null(tag))
    tag <- sprintf("%s%s_n%d_h%s_eps%s_ny%d_s%d",
                   if (signal) "alt" else "null",
                   if (signal) paste0("_", sig) else "",
                   n,
                   if (is.null(h_fix)) "LCV" else .fmt_num(h_fix),
                   .fmt_num(eps_mix), n_y_mult, seed)

  t_start <- Sys.time()
  if (cores > 1) suppressPackageStartupMessages(library(parallel))

  ## per-toy seeds: reproducible, order-independent (sequential or
  ## parallel give identical results), and any single toy replayable
  ## in isolation from its stored seed.
  set.seed(seed)
  toy_seeds <- sample.int(.Machine$integer.max, n_toys)

  run_one <- function(i) {
    set.seed(toy_seeds[i])
    d  <- gen_data(n = n, sig = sig, signal = signal)
    cv <- cvgof_fit_witness(d$x, K = K, n_y_mult = n_y_mult,
                            eps_mix = eps_mix, h = h_fix)
    r  <- cvgof_test(d$x, cv)
    stopifnot(!is.null(r$raw$Z_min), !is.null(r$orth$Z_min))

    nclip <- unname(rowSums(vapply(cv$per_fold,
                                   function(f) f$n_clip, numeric(2))))
    data.frame(
      toy = i, seed = toy_seeds[i], n_sig = d$n_sig, h = cv$h,
      Z_raw_X = r$raw$Z_X,     Z_orth_X = r$orth$Z_X,
      Z_raw_Y = r$raw$Z_Y,     Z_orth_Y = r$orth$Z_Y,
      Z_raw_min = r$raw$Z_min, Z_orth_min = r$orth$Z_min,
      tau_raw = r$raw$tau,     tau_orth = r$orth$tau,
      V_raw_X = r$V_tab["X","raw"],  V_orth_X = r$V_tab["X","orth"],
      V_raw_Y = r$V_tab["Y","raw"],  V_orth_Y = r$V_tab["Y","orth"],
      sandwich_X = r$sandwich_X, sandwich_Y = r$sandwich_Y,
      g_mean = mean(r$g_hat), g_sd = sd(r$g_hat),
      n_clip_x = nclip[1], n_clip_y = nclip[2]
    )
  }

  meta <- list(signal = signal,
               sig = if (signal) sig else NA_character_,
               n = n, h_fix = h_fix, eps_mix = eps_mix,
               n_y_mult = n_y_mult, n_toys = n_toys, K = K, seed = seed,
               cores = cores, tag = tag,
               lambda = if (signal) sg$lambda else NA_real_,
               started = t_start,
               sessionInfo = utils::sessionInfo())

  if (verbose) {
    cat(sprintf("\ntag=%s | signal=%s sig=%s n=%d K=%d n_toys=%d seed=%d\n",
                tag, signal, if (signal) sig else "-", n, K, n_toys, seed))
    cat(sprintf("h=%s eps_mix=%.3g n_y_mult=%d cores=%d\n",
                if (is.null(h_fix)) "LCV" else format(h_fix),
                eps_mix, n_y_mult, cores))
    flush.console()
  }

  out <- .toy_loop(run_one, n_toys, meta, tag, save_dir, chunk, cores,
                   verbose, overwrite, t_start)
  if (verbose) print(out)
  invisible(out)
}

## ---------------------------------------------------------------
## Shared chunked / checkpointed toy runner. run_one(i) returns a
## one-row data.frame (or errors, which become warnings and a dropped
## row). Saves <tag>.RData after every chunk and <tag>.csv at the end.
## Used by run_cvgof_experiment() (KDE) and run_tilt_experiment().
## ---------------------------------------------------------------
.toy_loop <- function(run_one, n_toys, meta, tag, save_dir, chunk, cores,
                      verbose, overwrite, t_start = Sys.time(),
                      class = "cvgof_experiment") {
  chunks <- split(seq_len(n_toys), ceiling(seq_len(n_toys) / chunk))
  acc  <- vector("list", length(chunks))
  t0   <- Sys.time()
  fout <- file.path(save_dir, paste0(tag, ".RData"))
  if (file.exists(fout) && !isTRUE(overwrite))
    stop(".toy_loop: ", fout, " exists; pass overwrite = TRUE")

  for (ci in seq_along(chunks)) {
    safe <- function(i) tryCatch(run_one(i), error = function(e) {
      warning("toy ", i, ": ", conditionMessage(e)); NULL })
    out_c <- if (cores > 1) parallel::mclapply(chunks[[ci]], safe, mc.cores = cores)
             else lapply(chunks[[ci]], safe)
    acc[ci] <- list(do.call(rbind, out_c))   # acc[[ci]] <- NULL would DELETE

    if (verbose) {
      done <- sum(vapply(acc, NROW, integer(1)))
      el   <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
      cat(sprintf("[%s] chunk %d/%d | %d/%d | %.1f min | eta %.1f min\n",
                  format(Sys.time()), ci, length(chunks), done, n_toys,
                  el, el * (n_toys - done) / max(done, 1)))
      flush.console()
    }
    ## checkpoint: a killed job loses at most one chunk
    results <- do.call(rbind, acc)
    save(results, meta, file = fout)
  }

  results <- do.call(rbind, acc)
  meta$ended   <- Sys.time()
  meta$elapsed <- as.numeric(difftime(meta$ended, t_start, units = "mins"))
  save(results, meta, file = fout)
  write.csv(results, file.path(save_dir, paste0(tag, ".csv")),
            row.names = FALSE)

  structure(list(results = results, meta = meta, file = fout), class = class)
}

## Z columns of a results table, in display order. "Y" = simulated
## calibration draws (KDE pipeline), "Q" = quadrature (tilt pipeline).
.z_cols <- function(res) {
  cand <- c("Z_raw_X", "Z_orth_X", "Z_raw_Y", "Z_orth_Y", "Z_raw_Q", "Z_orth_Q",
            "Z_fit_X", "Z_fit_Q")
  cand[cand %in% names(res)]
}
## the null-side column letter and headline Z column ("fit" row = conditional
## variance + cross-fold fitting-randomness term, when present)
.null_col <- function(res) if ("Z_orth_Y" %in% names(res)) "Y" else "Q"
.headline <- function(res) if ("Z_fit_Q" %in% names(res)) "Z_fit_Q" else
 paste0("Z_orth_", .null_col(res))

print.cvgof_experiment <- function(x, ...) {
  res <- x$results; m <- x$meta; B <- nrow(res)
  alt <- isTRUE(m$signal)
  zc  <- .z_cols(res); nc <- .null_col(res); hz <- .headline(res)
  kde <- !is.null(res$h)                      # KDE pipeline has h, n_clip, Z_*_min

  cat(sprintf("\n=========== %s summary: %s ===========\n",
              if (alt) "ALT" else "null", m$tag))
  if (!is.null(m$mode))
    cat(sprintf("pipeline=%s mode=%s%s\n", m$pipeline, m$mode,
                if (identical(m$mode, "pois"))
                  sprintf(" N0=%d sigma_A=(%s, %s)", m$N0,
                          format(m$sigma_A[1]), format(m$sigma_A[2])) else ""))

  zsum <- vapply(res[zc], function(z)
    c(mean = mean(z), med = median(z), sd = sd(z),
      q90 = quantile(z, .90, names = FALSE),
      q95 = quantile(z, .95, names = FALSE),
      q99 = quantile(z, .99, names = FALSE),
      max = max(z)), numeric(7))
  print(round(t(zsum), 3))

  cat(sprintf("\n%s at nominal 0.100 / 0.050 / 0.010:\n",
              if (alt) "power" else "upper-tail rejection rates"))
  rate <- function(z) c(a10 = mean(z > qnorm(.90)),
                        a05 = mean(z > qnorm(.95)),
                        a01 = mean(z > qnorm(.99)))
  rt <- vapply(res[zc], rate, numeric(3))
  if (kde) rt <- cbind(rt, min_raw = rate(res$Z_raw_min), min_orth = rate(res$Z_orth_min))
  if (!is.null(res$Z_orthnum_Q)) rt <- cbind(rt, orthnum_Q = rate(res$Z_orthnum_Q))
  print(round(rt, 4))

  if (alt) {
    cat(sprintf("\nMC sd at power ~0.5 (B = %d): %.4f\n", B, sqrt(.25 / B)))
    cat(sprintf("median %s = %.3f (MC se ~ %.3f) | criterion 2.2-2.5\n", hz,
                median(res[[hz]]), 1.253 * sd(res[[hz]]) / sqrt(B)))
    cat("\nn_sig:\n"); print(table(res$n_sig))
    if (length(unique(res$n_sig)) > 1) {
      fit <- lm(stats::reformulate("n_sig", hz), data = res)
      cat(sprintf("dose-response: dZ/dn_sig = %+.4f (se %.4f) | R2 %.3f\n",
                  coef(fit)[2], summary(fit)$coef[2, 2],
                  summary(fit)$r.squared))
    }
  } else {
    cat(sprintf("\nMC sd at these rates (B = %d): %.4f / %.4f / %.4f\n", B,
                sqrt(.10 * .90 / B), sqrt(.05 * .95 / B), sqrt(.01 * .99 / B)))
  }

  V_raw_N <- res[[paste0("V_raw_", nc)]]; V_orth_N <- res[[paste0("V_orth_", nc)]]
  sw_N    <- res[[paste0("sandwich_", nc)]]
  if (kde)
    cat(sprintf("\nh: median %.4f | %d distinct | [%.4f, %.4f]\n",
                median(res$h), length(unique(res$h)), min(res$h), max(res$h)))
  cat(sprintf("\nsandwich: X %.4f (sd %.4f) | %s %.4f (sd %.4f)\n",
              mean(res$sandwich_X), sd(res$sandwich_X), nc, mean(sw_N), sd(sw_N)))
  cat(sprintf("column X/%s: raw %.4f | orth %.4f\n", nc,
              mean(res$V_raw_X / V_raw_N), mean(res$V_orth_X / V_orth_N)))

  if (kde) {
    cat(sprintf("g_hat: mean %+.5f | mean fold-sd %.5f | |g|/sigma %.4f\n",
                mean(res$g_mean), mean(res$g_sd),
                abs(mean(res$g_mean)) / sqrt(mean(V_raw_N))))
    cat(sprintf("n_clip: x %d | y %d | %.3f per toy (X side)\n",
                sum(res$n_clip_x), sum(res$n_clip_y), mean(res$n_clip_x)))
  } else {
    gcols <- grep("^g[0-9]+_mean$", names(res), value = TRUE)
    for (gc in gcols) {
      j <- sub("^g([0-9]+)_mean$", "\\1", gc)
      cat(sprintf("g%s: mean %+.4f (toy-sd %.4f) | mean fold-sd %.4f | b%s mean %+.4f\n",
                  j, mean(res[[gc]]), sd(res[[gc]]), mean(res[[paste0("g", j, "_sd")]]),
                  j, mean(res[[paste0("b", j, "_mean")]])))
    }
    if (!is.null(res$v_aux) && any(res$v_aux > 0))
      cat(sprintf("v_aux share of V_orth_Q + v_aux: %.4f\n",
                  mean(res$v_aux / (res$V_orth_Q + res$v_aux))))
    if (!is.null(res$S_orth))
            cat(sprintf("S_orth vs S: cor %.3f | sd ratio %.3f  (1 iff unconstrained)\n",
                  cor(res$S, res$S_orth), sd(res$S_orth) / sd(res$S)))
    if (!is.null(res$V_cross))
      cat(sprintf("cross-fold term: V_cross / V_orth_Q = %.3f (sd %.3f) | active dim %.1f | var(S)/mean(V_fit) = %.3f\n",
                  mean(res$V_cross / res$V_orth_Q), sd(res$V_cross / res$V_orth_Q),
                  mean(res$p_act), var(res$S) / mean(res$V_orth_Q + res$v_aux + res$V_cross)))
    cat(sprintf("active bumps: mean %.2f per fold | max %d | lambda_z %.3f | J %d\n",
                mean(res$n_active_mean), max(res$n_active_max),
                res$lambda_z[1], res$J[1]))
    if (!all(is.na(res$beta_tan_mean)))
      cat(sprintf("tangent coef (score atom): mean %+.4f (sd %.4f)\n",
                  mean(res$beta_tan_mean), sd(res$beta_tan_mean)))
    if (identical(m$mode, "pois"))
      cat(sprintf("nu_hat: n %+.4f (sd %.4f) | s %+.4f (sd %.4f) | N mean %.1f\n",
                  mean(res$nu_n_hat), sd(res$nu_n_hat),
                  mean(res$nu_s_hat), sd(res$nu_s_hat), mean(res$N)))
    cat(sprintf("KKT max violation: %.2e\n", max(res$kkt_max)))
  }
  cat(sprintf("\nelapsed %.1f min (%.1f s/toy) | %s\n",
              m$elapsed, 60 * m$elapsed / B, x$file))
  invisible(x)
}

## =============================================================
## plot method: 2x2 studentization table vs the N(0,1) reference
## =============================================================
plot_z_hist <- function(z, main = "", bw = 0.4, xlim = NULL,
                        ref_sd = c(0.9, 1.1), fitted = TRUE,
                        show_ks = TRUE, legend = TRUE) {
  B <- length(z)
  if (is.null(xlim)) xlim <- c(min(-4, floor(min(z))), max(4, ceiling(max(z))))

  brk  <- seq(floor(xlim[1] / bw) * bw, ceiling(xlim[2] / bw) * bw, by = bw)
  hh   <- hist(z, breaks = brk, plot = FALSE)
  dens <- hh$counts / (B * bw)
  ## Poisson error on the bin count. The binomial correction
  ## (1 - p_bin) is < 7% here even in the modal bin, so it is left out;
  ## empty bins get a zero bar rather than a one-sided interval.
  se   <- sqrt(hh$counts) / (B * bw)

  ymax <- max(c(dens + se, dnorm(0))) * 1.18
  plot(NA, xlim = xlim, ylim = c(0, ymax), las = 1,
       xlab = "Z", ylab = "density", main = main)

  rect(head(hh$breaks, -1), 0, tail(hh$breaks, -1), dens,
       col = "#a6cee3", border = "#4a90b8")
  arrows(hh$mids, pmax(dens - se, 0), hh$mids, dens + se,
         angle = 90, code = 3, length = 0.02, col = "#1f5f7a")

  zz <- seq(xlim[1], xlim[2], length.out = 512)
  for (s in ref_sd) lines(zz, dnorm(zz, 0, s), lty = 2, lwd = 1.4, col = "#66c2a5")
  if (fitted) lines(zz, dnorm(zz, mean(z), sd(z)), lty = 3, lwd = 2, col = "#d95f02")
  lines(zz, dnorm(zz), lwd = 3, col = "#2b3a8f")

  if (show_ks) {
    ## Fully specified null, so the KS reference distribution is exact
    ## -- no Lilliefors correction needed.
    p <- suppressWarnings(ks.test(z, "pnorm")$p.value)
    legend("topleft", bty = "n", cex = 0.85,
           legend = c(sprintf("B = %d", B),
                      sprintf("mean %+.3f  sd %.3f", mean(z), sd(z)),
                      sprintf("KS p = %.3g", p)))
  }
  if (legend) {
    lt <- c("N(0,1) (target)", sprintf("N(0, %.1f^2)", ref_sd))
    lc <- c("#2b3a8f", rep("#66c2a5", length(ref_sd)))
    ll <- c(1, rep(2, length(ref_sd)));  lw <- c(3, rep(1.4, length(ref_sd)))
    if (fitted) { lt <- c(lt, "fitted normal"); lc <- c(lc, "#d95f02")
                  ll <- c(ll, 3); lw <- c(lw, 2) }
    legend("topright", legend = lt, col = lc, lty = ll, lwd = lw,
           bty = "n", cex = 0.8)
  }
  invisible(hh)
}

plot.cvgof_experiment <- function(x,
                                  which = NULL,
                                  file  = NULL,
                                  width = 10, height = 8,
                                  bw = 0.4, xlim = NULL, ...) {
  res <- x$results; m <- x$meta
  alt <- isTRUE(m$signal)
    zc  <- .z_cols(res)
  if (is.null(which) || identical(which, "all"))
    which <- if ("Z_fit_Q" %in% zc) c("Z_raw_Q", "Z_orth_Q", "Z_fit_X", "Z_fit_Q") else zc
  which <- match.arg(which, c(zc, "Z_orthnum_Q"), several.ok = TRUE)
  lab <- c(Z_raw_X = "raw x X", Z_orth_X = "orth x X",
           Z_raw_Y = "raw x Y", Z_orth_Y = "orth x Y",
           Z_raw_Q = "raw x Q", Z_orth_Q = "orth x Q",
           Z_fit_X = "fit (+cross) x X", Z_fit_Q = "fit (+cross) x Q",
           Z_orthnum_Q = "orth-numerator x Q")[which]
  if (is.null(xlim)) {
    xl <- range(unlist(res[which]))
    xlim <- c(min(-4, floor(xl[1])), max(4, ceiling(xl[2])))
  }
  n_pan <- length(which)

  ## Device. The two legends anchor independently at topleft/topright,
  ## so a narrow panel lets the right box overflow into the left one.
  ## Single-panel calls therefore get a width floor; multi-panel keeps
  ## the caller's width/height.
  if (!is.null(file)) {
    if (isTRUE(file)) file <- paste0(m$tag, "_hist.pdf")
    if (n_pan == 1) { width <- max(width, 7.5); height <- max(height, 6) }
    pdf(file, width = width, height = height)
    on.exit(dev.off(), add = TRUE)
  } else if (n_pan == 1 && (dev.cur() == 1L || par("din")[1] < 6.5)) {
    dev.new(width = 7.5, height = 6)
  }

  op <- par(mar = c(4.2, 4.4, 2.6, 1.2), mgp = c(2.6, 0.8, 0),
            mfrow = if (n_pan == 1) c(1, 1)
                    else if (n_pan == 2) c(1, 2) else c(2, 2),
            oma = if (n_pan > 1) c(0, 0, 2.2, 0) else c(0, 0, 0, 0))
  on.exit(par(op), add = TRUE)

  for (j in seq_along(which))
    plot_z_hist(res[[which[j]]], main = lab[j], xlim = xlim, bw = bw,
                show_ks = !alt, ...)

  if (n_pan > 1)
    mtext(sprintf("%s  (%s, B = %d)",
                  m$tag, if (alt) "alternative" else "null", nrow(res)),
          outer = TRUE, cex = 1.0, font = 2)
  invisible(x)
}

#### The following is the hybrid penalized log-density ratio estimator, designed
#### to catch the signal types in both NP1 and NP2.
## =============================================================
## Segment 1 (general form): hybrid dictionary under a fitted null
## =============================================================
## The dictionary is a function of the fitted null p_theta (density,
## score, quantile function), rebuilt per fold. Nothing assumes the
## exponential family or theta = 1; the exponential case is a check.
##
## Atoms, all orthonormal under p_theta by quadrature:
##   smooth block  W = span{P_1..P_m}, Legendre-type polynomials in
##                 2F_theta(x)-1, m = n_smooth + d0. The block is an
##                 orthonormal basis of W ∩ s_theta^⊥, built so atom k
##                 lies in span(P_1..P_{k+d0}). Bounded, exactly
##                 orthogonal to {1, s}.
##   bump block    Gaussians projected against {1, smooth block}, then
##                 made orthogonal to s by subtracting a multiple of
##                 R = normalized projection of s onto W (bounded), not
##                 of s itself. Scaled to unit variance.
##   No atom carries an unbounded component; the tilt is integrable
##   for every beta.
##
## Quadrature: trapezoid on a uniform grid of n_grid nodes over
## [q(tail_prob), q(1 - tail_prob)] (finite support ends replace the
## quantile). The same grid carries the atoms for glmnet's Lindsey/
## Poisson rows, the partition function Z(beta), and the interpolated
## data counts.

## ---------------------------------------------------------------
## Null-family interface. Each family supplies closures of theta.
## ---------------------------------------------------------------
null_exponential <- function() {
  list(
    name     = "exponential (rate)",
    d0       = 1L,
    mle      = function(x) 1 / mean(x),
    dens     = function(x, theta) dexp(x, rate = theta),
    cdf      = function(x, theta) pexp(x, rate = theta),
    score    = function(x, theta) cbind(1 / theta - x),
    quantile = function(p, theta) qexp(p, rate = theta),
    rand     = function(m, theta) rexp(m, rate = theta),
    support  = c(0, Inf)
  )
}

## ---------------------------------------------------------------
## Weighted Gram-Schmidt: orthonormalize columns of M under weights w,
## optionally against an already-orthonormal block Q. Two passes for
## stability. Columns whose residual norm falls below `tol` (relative
## to their own norm) are dropped as linearly dependent.
## ---------------------------------------------------------------
wgs <- function(M, w, Q = NULL, tol = 1e-8) {
  ip <- function(a, b) sum(a * b * w)
  keep <- integer(0); out <- NULL
  for (j in seq_len(ncol(M))) {
    v <- M[, j]; nv0 <- sqrt(ip(v, v))
    for (pass in 1:2) {
      if (!is.null(Q))   for (i in seq_len(ncol(Q)))   v <- v - ip(v, Q[, i])   * Q[, i]
      if (!is.null(out)) for (i in seq_len(ncol(out))) v <- v - ip(v, out[, i]) * out[, i]
    }
    nv <- sqrt(ip(v, v))
    if (nv > tol * nv0) { out <- cbind(out, v / nv); keep <- c(keep, j) }
  }
  list(Q = out, keep = keep)
}

## ---------------------------------------------------------------
## Quadrature grid under the fitted null: trapezoid on a uniform grid
## over [q(tail_prob), q(1 - tail_prob)] (finite support ends replace
## the quantile). Shared by make_dict() and by the fixed-witness checks
## of the Poisson extension, so the two never disagree on the grid.
## ---------------------------------------------------------------
make_quad_grid <- function(null, theta, n_grid = 16001L, tail_prob = 1e-9) {
  stopifnot(n_grid >= 50L)
  lo <- if (is.finite(null$support[1])) null$support[1] else null$quantile(tail_prob, theta)
  hi <- if (is.finite(null$support[2])) null$support[2] else null$quantile(1 - tail_prob, theta)
  x_grid <- seq(lo, hi, length.out = n_grid)
  dx     <- x_grid[2] - x_grid[1]
  w_grid <- rep(dx, n_grid); w_grid[c(1, n_grid)] <- dx / 2
  p0     <- null$dens(x_grid, theta)
  list(x_grid = x_grid, dx = dx, w_grid = w_grid, p0 = p0, wp = w_grid * p0)
}

## ---------------------------------------------------------------
## Dictionary constructor
## ---------------------------------------------------------------
## tangent = TRUE appends the orthonormalized score direction(s) sQ as
## UNPENALIZED atoms (Extension section: the tangent atoms {1, s}). The
## constant tangent atom is glmnet's intercept and is never a column;
## sQ is the score part. Column order is [tangent | smooth | bump] and
## the index vectors idx_tan / idx_smooth / idx_bump are the only way
## downstream code should address blocks. With tangent = FALSE (the
## Module-3 default) the columns are identical to the pre-extension
## dictionary.
make_dict <- function(null, theta,
                      n_smooth    = 5L,
                      width_rel   = 0.2,
                      spacing_rel = 0.1,
                      bump_lo     = 0,
                      bump_hi     = 0.999,
                      tail_prob   = 1e-9,
                      n_grid      = 16001L,
                      tangent     = FALSE) {
 
  stopifnot(n_smooth >= 1L, width_rel > 0, spacing_rel > 0,
            bump_hi > bump_lo, n_grid >= 50L)
 
  ## --- quadrature grid ------------------------------------------
  qg     <- make_quad_grid(null, theta, n_grid = n_grid, tail_prob = tail_prob)
  x_grid <- qg$x_grid; dx <- qg$dx; w_grid <- qg$w_grid; p0 <- qg$p0; wp <- qg$wp
  mass   <- sum(wp)
  E      <- function(f) sum(f * wp)
  mu     <- E(x_grid)
  sd0    <- sqrt(E((x_grid - mu)^2))                    # bump scale
 
  ## --- tangent directions: constant and centered score ----------
  one <- wgs(cbind(rep(1, n_grid)), wp)$Q               # normalized constant
  s_raw <- null$score(x_grid, theta)
  sQ  <- wgs(s_raw, wp, Q = one)$Q                      # orthonormal score dirs
  d0  <- ncol(sQ)
  ## off-grid map for sQ: sQ = [1, s] %*% tan_coef exactly (linear)
  tan_coef <- qr.solve(cbind(1, s_raw), sQ)
 
  ## --- bounded polynomial span W in F --------------------------
  Fg   <- null$cdf(x_grid, theta)
  m    <- n_smooth + d0
  degs <- seq_len(m + 2L)                               # a few spares
  pl   <- wgs(outer(2 * Fg - 1, degs, "^"), wp, Q = one)
  if (length(pl$keep) < m)
    stop("make_dict: only ", length(pl$keep), " independent polynomial atoms")
  Qp    <- pl$Q[, seq_len(m), drop = FALSE]             # orthonormal, ⊥ 1
  pkeep <- pl$keep[seq_len(m)]
 
  ## --- smooth block: hierarchical basis of W ∩ s^⊥ ---------------
  Gs <- crossprod(Qp, sQ * wp)                          # <P_k, s_l>, m x d0
  V  <- matrix(0, m, 0)
  for (k in (d0 + 1L):m) {
    idx <- seq_len(k)
    Gk  <- Gs[idx, , drop = FALSE]
    v   <- numeric(m); v[k] <- 1
    v[idx] <- v[idx] - as.vector(Gk %*% solve(crossprod(Gk), crossprod(Gk, v[idx])))
    for (pass in 1:2) if (ncol(V)) v <- v - as.vector(V %*% crossprod(V, v))
    V <- cbind(V, v / sqrt(sum(v^2)))
  }
  Qs <- Qp %*% V                                        # n_grid x n_smooth
  smooth_deg <- pkeep[(d0 + 1L):m]                      # highest degree in each atom
 
  ## bounded score surrogate: projection of s onto W, orthonormalized
  Rm <- Qp %*% qr.Q(qr(Gs))                             # n_grid x d0
  GR <- crossprod(Rm, sQ * wp)                          # <R_i, s_l>, d0 x d0
  score_in_W <- sqrt(colSums(Gs^2))                     # |proj_W s_l|, ~1 if W ≈ captures s
 
  ## --- bump block -----------------------------------------------
  width   <- width_rel   * sd0
  spacing <- spacing_rel * sd0
  ## bump range: finite support ends where available, else quantiles;
  ## with the default bump_lo = 0 and infinite lower support, mirror bump_hi
  q_lo <- if (bump_lo == 0 && !is.finite(null$support[1])) 1 - bump_hi else bump_lo
  c_lo <- if (is.finite(null$support[1]) && bump_lo == 0) null$support[1]
          else null$quantile(q_lo, theta)
  c_hi <- if (is.finite(null$support[2]) && bump_hi == 1) null$support[2]
          else null$quantile(bump_hi, theta)
  centers <- seq(c_lo, c_hi, by = spacing)
  J       <- length(centers)
  Braw    <- exp(-outer(x_grid, centers, "-")^2 / (2 * width^2))
 
  Q1s   <- cbind(one, Qs)                               # orthonormal
  C1    <- crossprod(Q1s, Braw * wp)                    # (1+n_smooth) x J
  gB    <- crossprod(sQ, Braw * wp)                     # d0 x J   <B_j, s>
  Dm    <- solve(t(GR), gB)                             # d0 x J
  Bproj <- Braw - Q1s %*% C1 - Rm %*% Dm                # ⊥ 1, ⊥ Qs, ⊥ s
  sd_B  <- sqrt(colSums(Bproj^2 * wp))
  if (any(sd_B < 1e-6 * sqrt(colSums(Braw^2 * wp))))
    warning("make_dict: near-degenerate bump atom(s)")
  Bstd  <- sweep(Bproj, 2, sd_B, "/")
 
  ## --- assemble: [tangent | smooth | bump] ------------------------
  p_tan  <- if (tangent) d0 else 0L
  A_grid <- if (tangent) cbind(sQ, Qs, Bstd) else cbind(Qs, Bstd)
  colnames(A_grid) <- c(if (tangent) paste0("S", seq_len(d0)),
                        paste0("P", seq_len(n_smooth)), paste0("B", seq_len(J)))
  pen_factor <- c(rep(0, p_tan), rep(0, n_smooth), rep(1, J))
  idx_tan    <- seq_len(p_tan)
  idx_smooth <- p_tan + seq_len(n_smooth)
  idx_bump   <- p_tan + n_smooth + seq_len(J)
 
  structure(list(
    theta = theta, null_name = null$name, d0 = d0,
    x_grid = x_grid, dx = dx, w_grid = w_grid, p0 = p0, wp = wp, mass = mass,
    mu = mu, sd0 = sd0,
    A_grid = A_grid, pen_factor = pen_factor,
    tangent = tangent, p_tangent = p_tan, p_smooth = n_smooth, J = J,
    idx_tan = idx_tan, idx_smooth = idx_smooth, idx_bump = idx_bump,
    idx_unpen = c(idx_tan, idx_smooth),
    smooth_deg = smooth_deg,
    centers = centers, width = width, spacing = spacing,
    score_in_W = score_in_W,
    ## off-grid re-evaluation: all corrections live in span{1, (2F-1)^d};
    ## the tangent atoms live in span{1, s}
    Qproj_grid = cbind(one, Qs, Rm), P_coef = rbind(C1, Dm),
    tan_coef = tan_coef,
    sd_B = sd_B, degs = degs, pkeep = pkeep
  ), class = "hybrid_dict")
}

## ---------------------------------------------------------------
## Off-grid evaluation: [1, Qs, R] are linear combinations of the raw
## columns [1, (2F-1)^d]; recover the map by least squares on the grid
## (exact), apply at new x, then rebuild the bumps.
## ---------------------------------------------------------------
eval_atoms <- function(dict, null, x) {
  rawcols <- function(xx) {
    Fx <- null$cdf(xx, dict$theta)
    cbind(1, outer(2 * Fx - 1, dict$degs, "^")[, dict$pkeep, drop = FALSE])
  }
  Cmat    <- qr.solve(rawcols(dict$x_grid), dict$Qproj_grid)
  Qproj_x <- rawcols(x) %*% Cmat
  Qs_x    <- Qproj_x[, 1L + seq_len(dict$p_smooth), drop = FALSE]
  Braw_x  <- exp(-outer(x, dict$centers, "-")^2 / (2 * dict$width^2))
  Bstd_x  <- sweep(Braw_x - Qproj_x %*% dict$P_coef, 2, dict$sd_B, "/")
  A <- if (isTRUE(dict$tangent)) {
    sQ_x <- cbind(1, null$score(x, dict$theta)) %*% dict$tan_coef
    cbind(sQ_x, Qs_x, Bstd_x)
  } else cbind(Qs_x, Bstd_x)
  colnames(A) <- colnames(dict$A_grid)
  A
}

## ---------------------------------------------------------------
## Linear-interpolation counts on the grid (for glmnet's Poisson rows).
## sum_g c_g A(x_g) matches sum_i A(x_i) exactly for linear A and to
## O(dx^2 A'') otherwise. Points outside the grid are clipped (warning).
## ---------------------------------------------------------------
interp_counts <- function(dict, x) {
  G <- length(dict$x_grid)
  out_of <- x < dict$x_grid[1] | x > dict$x_grid[G]
  if (any(out_of)) {
    warning("interp_counts: ", sum(out_of), " point(s) outside grid; clipped")
    x <- pmin(pmax(x, dict$x_grid[1]), dict$x_grid[G])
  }
  pos  <- (x - dict$x_grid[1]) / dict$dx
  g0   <- pmin(floor(pos), G - 2L)
  frac <- pos - g0
  acc  <- rowsum(c(1 - frac, frac), c(g0 + 1L, g0 + 2L))
  cnt  <- numeric(G)
  cnt[as.integer(rownames(acc))] <- acc[, 1]
  cnt
}


## ---------------------------------------------------------------
## Self-check
## ---------------------------------------------------------------
check_dict <- function(dict, null, x_test = NULL) {
  ip  <- function(F, Gm) crossprod(F, Gm * dict$wp)
  A   <- dict$A_grid
  Gm  <- ip(A, A)
  tan <- cbind(1, null$score(dict$x_grid, dict$theta))
  is  <- dict$idx_smooth; ib <- dict$idx_bump; it <- dict$idx_tan
  ps  <- dict$p_smooth
  nt  <- c(is, ib)                             # non-tangent atoms
  jm  <- ib[dict$J %/% 2]                      # an interior bump
  G   <- nrow(A)
  out <- list(
    mass            = dict$mass,                         # ~ 1
    atom_means      = max(abs(colSums(A * dict$wp))),    # ~ 0
    gram_vs_I       = max(abs(Gm - diag(ncol(A)))[c(it, is), , drop = FALSE]), # ~ 0
    atom_unit_var   = max(abs(diag(Gm) - 1)),            # ~ 0
    atoms_vs_score  = max(abs(ip(tan, A[, nt, drop = FALSE]))),  # ~ 0
    tangent_vs_score= if (length(it)) max(abs(ip(tan, A[, it, drop = FALSE]))) else NA, # ~ 1
    smooth_vs_bump  = max(abs(Gm[is, ib])),              # ~ 0
    adj_bump_corr   = Gm[jm, jm + 1L],                   # ~ 0.92
    smooth_max_abs  = max(abs(A[, is])),                 # ~ 3-4
    bump_max_abs    = max(abs(A[, ib])),                 # ~ 50
    tail_integrand  = max(A[G, ]^2 * dict$wp[G]),        # << 1e-8
    score_in_W      = dict$score_in_W,                   # ~ 0.99
    smooth_degrees  = dict$smooth_deg, J = dict$J, p = ncol(A),
    p_tangent = dict$p_tangent,
    n_grid = G, dx = dict$dx,
    width = dict$width, spacing = dict$spacing, range = range(dict$centers)
  )
  if (!is.null(x_test)) {
    A_x  <- eval_atoms(dict, null, x_test)
    cnt  <- interp_counts(dict, x_test)
    err  <- (colSums(A_x) - colSums(A * cnt)) / sqrt(length(x_test))
    out$offgrid_vs_grid = max(abs(eval_atoms(dict, null, dict$x_grid) - A))  # ~ 1e-11
    out$count_total     = c(binned = sum(cnt), n = length(x_test))          # equal
    out$mle_binned_vs_data = c(binned = sum(cnt) / sum(cnt * dict$x_grid),
                               data   = null$mle(x_test))                   # equal (Exp)
    out$suffstat_err    = max(abs(err))                  # ~ 1.6e-3 at n_grid 16001
    out$suffstat_worst  = colnames(A)[which.max(abs(err))]   # B1
    out$suffstat_smooth = max(abs(err[is]))              # ~ 8e-4
  }
  out
}

## Example:
##   nul  <- null_exponential()
##   d1   <- make_dict(nul, theta = 1)
##   str(check_dict(d1, nul, x_test = rexp(1800)))
##   ## scale equivariance: same geometry at theta = 0.86
##   d2   <- make_dict(nul, theta = 0.86)
##   str(check_dict(d2, nul, x_test = rexp(1800, 0.86)))


## =============================================================
## Segment 2: penalized exponential tilt via glmnet (Lindsey/Poisson)
## =============================================================
## Model:  n_beta(x) = expo * p_theta(x) exp(a + f_beta(x)),  f_beta = A(x) beta,
##         A = [tangent | smooth | bump] (tangent block empty in cond mode)
## Poisson rows on the quadrature grid:
##         y_g = interpolated counts, offset_g = log(expo * wp_g),
##         mu_g = expo * wp_g * exp(a + A_g beta)
## glmnet objective (nobs = n_grid rows, unit weights):
##         -(1/N) sum_g (y_g eta_g - e^eta_g) + lambda sum_j pf_j |beta_j|
## KKT at beta_j = 0:  |(1/N) sum_g (y_g - mu_g) A_gj| <= lambda pf_j
## With z_j = sum_g (y_g - mu_g) A_gj / sqrt(expo) the standardized score,
##         lambda_glmnet = lambda_z * sqrt(expo) / N.
##
## The two modes differ ONLY in `expo` and in which constant defines
## the witness:
##   cond (iid, Module 3):  expo = n_tr, the sample size. The intercept
##        a is Lindsey's normalizer, a = -log Z(beta) at the optimum,
##        and the witness is log(p_beta/p_theta) = f_beta - log Z with
##        log Z recomputed by quadrature (normalize = TRUE).
##   pois (Extension):      expo = (K-1)/K * N0 e^{nu_n}, the training
##        exposure of the FITTED null intensity. There is no free
##        normalization: a is the coefficient of the constant tangent
##        atom (the fitted log-excess in expected events) and the witness
##        is the log intensity ratio a + f_beta (normalize = FALSE).
##        a + log Z = log(N_tr / expo) at the optimum (GLM total-matching).

lambda_to_glmnet <- function(lambda_z, expo, n_grid) lambda_z * sqrt(expo) / n_grid

## ---------------------------------------------------------------
## Fit. lambda_z is the penalty on the bump block in z-units; tangent
## and smooth blocks are unpenalized. Returns beta, the intercept, log Z
## by quadrature, the witness constant, and the witness on the grid.
## ---------------------------------------------------------------
fit_tilt <- function(dict, null, x_tr, lambda_z,
                     expo = NULL, normalize = TRUE,
                     thresh = 1e-10, maxit = 1e5, n_path = 5L) {
  if (!requireNamespace("glmnet", quietly = TRUE))
    stop("fit_tilt: package 'glmnet' is required")
  n  <- length(x_tr)
  if (is.null(expo)) expo <- n                    # cond mode default
  stopifnot(expo > 0)
  N  <- length(dict$x_grid)
  y  <- interp_counts(dict, x_tr)
  off <- log(expo * dict$wp)
  lam <- lambda_to_glmnet(lambda_z, expo, N)

  ## glmnet is unreliable with a single lambda; give it a short path
  ## ending at the target and take the last solution (warm-started).
  lam_path <- lam * exp(seq(log(4), 0, length.out = n_path))

  fit <- glmnet::glmnet(x = dict$A_grid, y = y, family = "poisson",
                        offset = off, penalty.factor = dict$pen_factor,
                        lambda = lam_path, standardize = FALSE,
                        intercept = TRUE, thresh = thresh, maxit = maxit)
  beta <- as.vector(fit$beta[, n_path])
  a    <- unname(fit$a0[n_path])

  ## partition function by quadrature (exact for the discretized model)
  f_grid <- as.vector(dict$A_grid %*% beta)
  logZ   <- log(sum(dict$wp * exp(f_grid)))
  const  <- if (normalize) -logZ else a
  phi_grid <- f_grid + const

  ## profiled standardized scores at the solution (KKT diagnostics)
  mu   <- expo * dict$wp * exp(a + f_grid)
  z    <- as.vector(crossprod(dict$A_grid, y - mu)) / sqrt(expo)
  ib   <- dict$idx_bump

  ## the score tangent atom is unbounded (linear in x for the exponential
  ## null); the tilt is integrable off-grid only for |beta_S| < 1 in the
  ## standardized parameterization. Nowhere near it in practice.
  if (length(dict$idx_tan) && any(abs(beta[dict$idx_tan]) > 0.9))
    warning("fit_tilt: |beta| on a tangent atom > 0.9; tilt near non-integrable")

  structure(list(
    beta = beta, a0 = a, logZ = logZ, const = const, normalize = normalize,
    phi_grid = phi_grid, f_grid = f_grid,
    lambda_z = lambda_z, lambda_glmnet = lam, expo = expo,
    active_bumps = which(beta[ib] != 0),
    beta_tan = beta[dict$idx_tan],
    z_at_fit = z, n = n,
    ## cond: a + logZ ~ 0 at convergence. pois: = log(N_tr / expo), the
    ## log ratio of observed to fitted-null-expected training count.
    intercept_vs_logZ = a + logZ,
    log_count_ratio   = log(n / expo),
    max_kkt_violation = max(abs(z[ib]) - lambda_z, 0),   # ~ 0 (bumps at bound)
    dev_ratio = fit$dev.ratio[n_path], npasses = fit$npasses
  ), class = "tilt_fit")
}

## Witness at arbitrary x: cond  log(p_beta / p_theta) = f_beta(x) - log Z
##                         pois  log(n_w / n_0)        = a + f_beta(x)
## Both are f_beta(x) + fit$const.
phi_tilt <- function(fit, dict, null, x) {
  as.vector(eval_atoms(dict, null, x) %*% fit$beta) + fit$const
}

## ---------------------------------------------------------------
## Profiled entry threshold: max_j |z_j| over the bump block at the
## unpenalized-only fit (tangent + smooth). This is the exact lambda_z at
## which the first bump enters. Used for the null pre-simulation of the
## penalty. `expo` as in fit_tilt().
## ---------------------------------------------------------------
entry_threshold <- function(dict, null, x_tr, thresh = 1e-10, expo = NULL) {
  n   <- length(x_tr); N <- length(dict$x_grid)
  if (is.null(expo)) expo <- n
  y   <- interp_counts(dict, x_tr)
  off <- log(expo * dict$wp)
  iu  <- dict$idx_unpen; ib <- dict$idx_bump
  ## unpenalized-only Poisson GLM (+ intercept): use glm for an exact,
  ## penalty-free fit
  df  <- data.frame(y = y, dict$A_grid[, iu, drop = FALSE])
  g   <- stats::glm(y ~ ., family = poisson(), data = df, offset = off,
                    control = glm.control(epsilon = thresh, maxit = 100))
  mu  <- fitted(g)
  z   <- as.vector(crossprod(dict$A_grid, y - mu)) / sqrt(expo)
  list(max_z_bump = max(abs(z[ib])), z = z,
       which = which.max(abs(z[ib])), beta_unpen = coef(g)[-1],
       beta_smooth = coef(g)[-1][match(dict$idx_smooth, iu)])
}

## Null pre-simulation: distribution of the entry threshold over B
## null training sets of size n_tr from p_theta_hat. Returns quantiles
## in z-units; pick lambda_z from these. the old cond-only, entry-threshold
## version (iid draws, plain MLE).
pretune_lambda <- function(null, theta, n_tr, B = 250L,
                           probs = c(.5, .8, .9, .95), seed = NULL, ...) {
  if (!is.null(seed)) set.seed(seed)
  dict <- make_dict(null, theta, ...)            # geometry fixed at theta
  mz <- vapply(seq_len(B), function(b) {
    x <- null$rand(n_tr, theta)
    ## refit theta on the simulated sample, as the pipeline will
    th_b <- null$mle(x)
    d_b  <- make_dict(null, th_b, ...)
    entry_threshold(d_b, null, x)$max_z_bump
  }, numeric(1))
  list(max_z = mz, quantiles = quantile(mz, probs),
       universal = sqrt(2 * log(dict$J)), J = dict$J, n_tr = n_tr)
}

## ---------------------------------------------------------------
## Self-checks
## ---------------------------------------------------------------
check_fit <- function(dict, null, x_tr, lambda_z = 3) {
  et  <- entry_threshold(dict, null, x_tr)
  eps <- 0.02
  f_above <- fit_tilt(dict, null, x_tr, et$max_z_bump * (1 + eps))
  f_below <- fit_tilt(dict, null, x_tr, et$max_z_bump * (1 - eps))
  f_lam   <- fit_tilt(dict, null, x_tr, lambda_z)
  ## witness sanity on training data and on fresh null draws
  y0 <- null$rand(20000, dict$theta)
  ph <- phi_tilt(f_lam, dict, null, y0)
  list(
    entry_z          = et$max_z_bump,
    entry_atom       = et$which,
    active_above     = length(f_above$active_bumps),   # 0: mapping exact
    active_below     = length(f_below$active_bumps),   # >= 1, and == et$which
    below_which      = f_below$active_bumps,
    intercept_vs_logZ= f_lam$intercept_vs_logZ,        # ~ 0
    kkt_violation    = f_lam$max_kkt_violation,        # ~ 0
    smooth_vs_glm    = max(abs(f_above$beta[dict$idx_smooth] - et$beta_smooth)),  # ~ 0
    n_active_at_lam  = length(f_lam$active_bumps),
    E_phi_null       = mean(ph),                       # <= 0 (KL(p_theta||p_beta) >= 0)
    E_expphi_null    = mean(exp(ph)),                  # ~ 1 (normalization)
    range_phi_null   = range(ph),
    npasses          = f_lam$npasses
  )
}

## Example:
##   nul <- null_exponential(); set.seed(2)
##   x   <- rexp(1809)                          # one training fold under H0
##   d   <- make_dict(nul, nul$mle(x))
##   str(check_fit(d, nul, x, lambda_z = 3))
##   ## an NP1 training fold: the spike should enter
##   xa  <- gen_data(n = 2010, sig = "NP1")$x[1:1809]
##   da  <- make_dict(nul, nul$mle(xa))
##   fa  <- fit_tilt(da, nul, xa, lambda_z = 3); fa$active_bumps; da$centers[fa$active_bumps]


## =============================================================
## Extension: Poisson observation model with auxiliary-constrained
## nuisances (Theory_Methods.md, "Extension"), run side by side with
## the iid ("conditional") version using the hybrid tilt as the fitter.
## =============================================================
## Two modes share one pipeline:
##   "cond"  fixed n, x iid, null = Exp(theta) density, plain MLE.
##           Modules 1-3 with the tilt witness and quadrature
##           calibration (no simulation draws). The KDE pipeline above
##           (cvgof_fit_witness / cvgof_test / run_cvgof_experiment) is
##           left untouched as the validated reference.
##   "pois"  N ~ Poisson, intensity n(x|R_nu) = N0 e^{nu_n} theta e^{-theta x},
##           theta = e^{-nu_s}; auxiliary measurement nu_A ~ N(nu*, Sigma_A),
##           Sigma_A = diag(sigma_n^2, sigma_s^2) (Inf = unconstrained).
##           Null fit = constrained extended-likelihood maximizer per
##           training fold; alternative = tilt with tangent atoms;
##           statistic = held-out extended log-likelihood ratio.
##
## The mode enters ONLY through the per-fold "frame" (fold_null_frame):
##   expo_te   expected test-fold event count under the fitted null
##             (cond: N_k, the realized fold size; pois: (1/K) N0 e^{nu_n})
##   expo_tr   training exposure (cond: n_tr; pois: (K-1)/K N0 e^{nu_n})
##   u(x)      score directions of the null family in the parameters
##             the nuisance fit moves along (cond: s_theta; pois:
##             d log n / d(nu_n, nu_s) = (1, -1 + theta x))
##   SigAinv   Sigma_A^{-1} (cond: 0)
## and through `centered` (cond: multinomial variance, pois: compound-
## Poisson uncentered variance) in cvgof_test_quad().
##
## Statistic and calibration (both modes), with omega_te = expo_te * wp
## the expected test-fold event weights on the quadrature grid:
##   S_k   = sum_{i in D_k} phi_k(X_i) - sum_g omega_te,g phi_k(x_g)
##   V_k^Q = sum_g omega_te,g phi_k^2  [- (sum_g omega_te,g phi_k)^2 / N_k if centered]
##   g_k   = sum_g omega_te,g phi_k(x_g) u(x_g)                       (d-vector)
##   I_k   = sum_g omega_tr,g u u^T + Sigma_A^{-1}
##   b_k   = (K-1) I_k^{-1} g_k
##   phi~_k = phi_k - u^T b_k                                     (orthogonalized)
## and the auxiliary-noise variance term (pois only)
##   v_aux = c^T Sigma_A^{-1} c,   c = sum_k I_k^{-1} g_k.
##
## NOTE on the (K-1): Theory_Methods.md (Extension) writes the
## orthogonalized witness as phi - g_k^T I_k^{-1} u with g_k the TEST-fold
## share (1/K) int n0 phi u and I_k the TRAINING-fold information. Each
## event enters K-1 training fits, so the per-event plug-in influence is
## (K-1) g_k^T I_k^{-1} u(x); the factor is what makes the Poisson formula
## reduce to (3.8) in the iid limit (g_k I_k^{-1} (K-1) = Cov(phi,s)/I
## for balanced folds). Verified in check_test_quad_fixed_witness().
## NOTE on v_aux: the auxiliary datum nu_A is redrawn per toy, so its
## fluctuation propagates through every hat-nu^{(-k)} into the plug-in
## term, independently of the events. It is not in the doc's variance.

N0_DEFAULT <- 2000

## ---------------------------------------------------------------
## Data generation, Poisson mode. Background N_b ~ Poisson(N0 e^{nu_n*}),
## x_b ~ Exp(e^{-nu_s*}); signal N_s ~ Poisson(N_i) from SIGNALS[[sig]]
## (superposition of independent processes, NPLM's additive n + N_i f_i;
## the signal shape does not move with nu*). nu_A ~ N(nu*, Sigma_A),
## coordinatewise; an infinite sigma leaves nu_A at nu* (unused).
## N_sig = m forces exactly m signal events (variance-reduced power).
## ---------------------------------------------------------------
gen_data_pois <- function(sig = SIG, nu_star = c(0, 0), sigma_A = c(0.1, 0.1),
                          N0 = N0_DEFAULT, signal = TRUE, N_sig = NULL) {
  sg <- SIGNALS[[sig]]
  if (is.null(sg)) stop("gen_data_pois: unknown sig '", sig, "'")
  stopifnot(length(nu_star) == 2L, length(sigma_A) == 2L, all(sigma_A > 0))
  theta_star <- exp(-nu_star[2])
  n_b <- rpois(1L, N0 * exp(nu_star[1]))
  n_s <- if (!signal) 0L else if (is.null(N_sig)) rpois(1L, sg$N_i) else as.integer(N_sig)
  x <- c(rexp(n_b, rate = theta_star), sg$r(n_s))
  z <- c(rep(0L, n_b), rep(1L, n_s))
  ord <- sample(length(x))
  nu_A <- nu_star
  fin  <- is.finite(sigma_A)
  if (any(fin)) nu_A[fin] <- rnorm(sum(fin), nu_star[fin], sigma_A[fin])
  names(nu_A) <- c("nu_n", "nu_s")
  list(x = x[ord], z = z[ord], N = length(x), n_sig = n_s,
       nu_A = nu_A, nu_star = nu_star, sigma_A = sigma_A, N0 = N0,
       theta_star = theta_star)
}

## ---------------------------------------------------------------
## Fold assignment. balanced = TRUE (both modes, the default): sizes
## differ by at most one. In cond mode this matches the CV-CLT
## bookkeeping. In pois mode it departs from the thinning construction
## of the Extension (each event uniform and independent, giving
## independent Poisson folds); the fitted-null calibration is unaffected
## because the per-fold expected counts Lambda/K still sum to Lambda and
## the per-fold variances to int n phi^2, whichever way N is split.
## Verified by check_test_quad_fixed_witness(balanced = TRUE): variance
## ratio 0.98, sd(Z) 0.99 at sigma_A = 0.1. balanced = FALSE remains
## available for the thinning construction.
## ---------------------------------------------------------------
make_folds <- function(n, K, balanced = TRUE) {
  if (balanced) sample(rep_len(seq_len(K), n)) else sample.int(K, n, replace = TRUE)
}

## ---------------------------------------------------------------
## Constrained null fit, Poisson mode. Maximizes
##   sum_{tr} log n(X_i|R_nu) - frac_tr * int n(.|R_nu) + ell_A(nu)
## which separates in the prototype into two concave 1-D problems:
##   nu_n:  N_tr - frac_tr N0 e^{nu_n} - (nu_n - nu_A,n)/sigma_n^2 = 0
##   nu_s: -N_tr + e^{-nu_s} sum x  - (nu_s - nu_A,s)/sigma_s^2 = 0
## Newton from the unconstrained roots (e^{nu_n} = N_tr/(frac_tr N0),
## e^{-nu_s} = 1/xbar); both F are monotone so the iteration is
## monotone after at most one overshoot. sigma = Inf gives weight 0.
## ---------------------------------------------------------------
fit_null_pois <- function(x_tr, frac_tr, nu_A, sigma_A, N0 = N0_DEFAULT,
                          tol = 1e-10, maxit = 50L) {
  N_tr <- length(x_tr); Sx <- sum(x_tr)
  stopifnot(N_tr > 0, frac_tr > 0, frac_tr <= 1, length(nu_A) == 2L)
  w  <- ifelse(is.finite(sigma_A), 1 / sigma_A^2, 0)
  c1 <- frac_tr * N0
  newton <- function(F, dF, v) {
    for (it in seq_len(maxit)) {
      f <- F(v); if (abs(f) < tol) break
      v <- v - f / dF(v)
    }
    c(v, it, F(v))
  }
  rn <- newton(function(v) N_tr - c1 * exp(v) - w[1] * (v - nu_A[1]),
               function(v) -c1 * exp(v) - w[1],
               log(N_tr / c1))
  rs <- newton(function(v) -N_tr + exp(-v) * Sx - w[2] * (v - nu_A[2]),
               function(v) -exp(-v) * Sx - w[2],
               log(Sx / N_tr))
  if (max(abs(rn[3]), abs(rs[3])) > 1e-6)
    warning("fit_null_pois: Newton residual ", signif(max(abs(rn[3]), abs(rs[3])), 3))
  nu <- c(nu_n = unname(rn[1]), nu_s = unname(rs[1]))
  list(nu = nu, theta = exp(-nu[[2]]),
       Lambda = N0 * exp(nu[[1]]),              # int n(.|R_nu_hat), full
       Lambda_tr = c1 * exp(nu[[1]]),           # training exposure
       N_tr = N_tr, iter = c(nu_n = unname(rn[2]), nu_s = unname(rs[2])),
       ## unconstrained roots, for the shrinkage diagnostic
       nu_free = c(nu_n = log(N_tr / c1), nu_s = log(Sx / N_tr)),
       weights = w)
}

## ---------------------------------------------------------------
## Per-fold frame: everything the test needs from the fitted null, in
## a mode-agnostic form (see the block comment above).
## Assumes theta = e^{-nu_s} in pois mode (log-scale nuisance), so the
## nu_s-direction of the intensity score is -theta * s_theta(x).
## ---------------------------------------------------------------
fold_null_frame <- function(mode, x_tr, N_k, K, null = null_exponential(),
                            nu_A = NULL, sigma_A = c(Inf, Inf), N0 = N0_DEFAULT) {
  n_tr <- length(x_tr)
  if (mode == "cond") {
    theta <- null$mle(x_tr)
    list(mode = mode, theta = theta, nu = c(nu_n = NA_real_, nu_s = -log(theta)),
         expo_tr = n_tr, expo_te = N_k, n_tr = n_tr, N_k = N_k,
         u_fun = function(x) null$score(x, theta),
         SigAinv = matrix(0, null$d0, null$d0),
         ## cond: the free MLE absorbs the tangent directions completely, so
         ## a tangent atom (if any) carries no fitting randomness: rho = 1
         shrink = c(nu_n = 1, nu_s = 1),
         normalize = TRUE, centered = TRUE, null_fit = NULL)
  } else if (mode == "pois") {
    stopifnot(!is.null(nu_A), null$d0 == 1L)
    nf <- fit_null_pois(x_tr, frac_tr = (K - 1) / K, nu_A = nu_A,
                        sigma_A = sigma_A, N0 = N0)
    th <- nf$theta
    ## Shrinkage of the constrained null fit toward nu_A, per direction, in
    ## the unit-information parameterization (count: Var = Lambda; score
    ## atom: unit variance): rho_d = Lambda_tr / (Lambda_tr + sigma_d^-2).
    ## The witness's tangent coefficient is (1 - rho_d)(nu_tilde - nu_A), so
    ## its fitting influence is damped by (1 - rho_d): 0 when unconstrained,
    ## 1 when pinned. Used by the cross-fold covariance term.
    w   <- ifelse(is.finite(sigma_A), 1 / sigma_A^2, 0)
    rho <- nf$Lambda_tr / (nf$Lambda_tr + w)
    list(mode = mode, theta = th, nu = nf$nu,
         expo_tr = nf$Lambda_tr, expo_te = nf$Lambda / K, n_tr = n_tr, N_k = N_k,
         u_fun = function(x) cbind(1, -th * as.vector(null$score(x, th))),
         SigAinv = diag(ifelse(is.finite(sigma_A), 1 / sigma_A^2, 0), 2L),
         shrink = c(nu_n = rho[1], nu_s = rho[2]),
         normalize = FALSE, centered = FALSE, null_fit = nf)
  } else stop("fold_null_frame: unknown mode '", mode, "'")
}

## ---------------------------------------------------------------
## Fold record from a frame, a dictionary/grid, and a witness given on
## the grid and at the held-out points. Shared by cvgof_fit_tilt() and
## by the fixed-witness check, so the test function sees one layout.
## ---------------------------------------------------------------
make_fold_record <- function(fr, x_grid, wp, idx_ev, x_ev, phi_x, phi_grid, extra = list()) {
  c(list(idx_ev = idx_ev, x_ev = x_ev, phi_x = phi_x, phi_grid = phi_grid,
         x_grid = x_grid,
         u_x = fr$u_fun(x_ev), u_grid = fr$u_fun(x_grid),
         omega_te = fr$expo_te * wp, omega_tr = fr$expo_tr * wp,
         N_k = fr$N_k, n_tr = fr$n_tr, theta = fr$theta, nu = fr$nu,
         expo_te = fr$expo_te, expo_tr = fr$expo_tr,
         SigAinv = fr$SigAinv, centered = fr$centered),
    extra)
}

## =============================================================
## Cross-fitted tilt witness (both modes)
## =============================================================
## x        events; K folds; mode "cond" or "pois"
## lambda_z bump penalty in z-units; NULL = universal sqrt(2 log J)
## nu_A, sigma_A, N0  pois-mode auxiliary measurement and reference
##          normalization (ignored in cond mode)
## tangent  include the score tangent atom; default TRUE in pois mode,
##          FALSE in cond mode (where the free nuisance absorbs it and
##          it would carry a pure noise degree of freedom)
## dict_args / fit_args  passed to make_dict() / fit_tilt()
## folds    explicit fold ids; default make_folds(balanced = TRUE) in both modes
cvgof_fit_tilt <- function(x, K = K_FOLDS, mode = c("cond", "pois"),
                           lambda_z = NULL, nu_A = NULL, sigma_A = c(Inf, Inf),
                           N0 = N0_DEFAULT, null = null_exponential(),
                           tangent = NULL, dict_args = list(), fit_args = list(),
                           folds = NULL, seed = NULL, bump_damping = c("none", "soft")) {
  mode <- match.arg(mode); bump_damping <- match.arg(bump_damping)
  if (!is.null(seed)) set.seed(seed)
  n <- length(x)
  if (is.null(tangent)) tangent <- (mode == "pois")
  if (mode == "pois") stopifnot(length(nu_A) == 2L, length(sigma_A) == 2L)
  if (is.null(folds)) folds <- make_folds(n, K, balanced = TRUE)
  stopifnot(length(folds) == n, all(folds %in% seq_len(K)))
  if (any(tabulate(folds, K) == 0L)) stop("cvgof_fit_tilt: empty fold")

  per_fold <- vector("list", K)
  ## Reference grid for the cross-fold covariance term: fold 1's quadrature
  ## grid and fitted-null weights. Every fold's active atoms are evaluated
  ## on it so that cross-Grams between folds are on one measure.
  x_ref <- wp_ref <- NULL
  for (k in seq_len(K)) {
    idx_ev <- which(folds == k)
    x_ev   <- x[idx_ev]
    x_tr   <- x[which(folds != k)]

    fr   <- fold_null_frame(mode, x_tr, N_k = length(idx_ev), K = K, null = null,
                            nu_A = nu_A, sigma_A = sigma_A, N0 = N0)
    dict <- do.call(make_dict, c(list(null = null, theta = fr$theta,
                                      tangent = tangent), dict_args))
    lz   <- if (is.null(lambda_z)) sqrt(2 * log(dict$J)) else lambda_z
    fit  <- do.call(fit_tilt, c(list(dict = dict, null = null, x_tr = x_tr,
                                     lambda_z = lz, expo = fr$expo_tr,
                                     normalize = fr$normalize), fit_args))
    phi_x <- phi_tilt(fit, dict, null, x_ev)

    ## --- fitting-randomness record (see cvgof_test_quad, cross term) ------
    ## Active design of the fitted GLM: [intercept (pois) | tangent | smooth |
    ## active bumps], on the reference grid, plus the fitted intensity there
    ## (Hessian weights) and the per-column influence damping (1 - rho) for
    ## the tangent directions. In cond mode the total count is fixed, so
    ## the intercept carries no randomness and is left out.
    if (k == 1L) { x_ref <- dict$x_grid; wp_ref <- dict$wp }
    A_ref  <- if (k == 1L) dict$A_grid else eval_atoms(dict, null, x_ref)
    i_act  <- c(dict$idx_tan, dict$idx_smooth, dict$idx_bump[fit$active_bumps])
    A_act  <- A_ref[, i_act, drop = FALSE]
    ## Damping of the active bumps in the cross-fold covariance.
    ## bump_damping = "none" (default): selected bumps count like free atoms.
    ## Inside its active set the lasso coefficient beta = (z - lambda sign z)/sqrt(expo)
    ## responds to the data with slope one, so its first-order influence on the
    ## other folds' statistics is the unpenalized one. Validated: 1-D null (this
    ## file's protocol) size 0.085/0.040/0.0125 at 0.10/0.05/0.01, mildly conservative
    ## (sd 0.91); 5-D null exactly calibrated (utils_5d.R, with the Reference term).
    ## bump_damping = "soft": the earlier heuristic factor (1 - lambda/|z|) =
    ## |beta| sqrt(expo) / (|beta| sqrt(expo) + lambda); calibrated in 1-D but
    ## anticonservative in 5-D (size 0.082 at 0.05). Kept for comparison.
    b_act  <- abs(fit$beta[dict$idx_bump[fit$active_bumps]]) * sqrt(fr$expo_tr)
    damp   <- c(rep(1 - fr$shrink[["nu_s"]], length(dict$idx_tan)),
                rep(1, length(dict$idx_smooth)),
                if (bump_damping == "none") rep(1, length(b_act)) else b_act / (b_act + lz))
    if (mode == "pois") {
      A_act <- cbind(1, A_act); damp <- c(1 - fr$shrink[["nu_n"]], damp)
    }
    mu_ref <- fr$expo_tr * wp_ref * exp(fit$a0 + as.vector(A_ref %*% fit$beta))

    per_fold[[k]] <- make_fold_record(
      fr, dict$x_grid, dict$wp, idx_ev, x_ev, phi_x, fit$phi_grid,
      extra = list(beta = fit$beta, a0 = fit$a0, logZ = fit$logZ, const = fit$const,
                   beta_tan = fit$beta_tan, active_bumps = fit$active_bumps,
                   centers_active = dict$centers[fit$active_bumps],
                   lambda_z = lz, J = dict$J, npasses = fit$npasses,
                   kkt = fit$max_kkt_violation,
                   nu_free = if (!is.null(fr$null_fit)) fr$null_fit$nu_free else NULL,
                   A_act_ref = A_act, mu_ref = mu_ref, damp = damp, shrink = fr$shrink))
  }

  list(per_fold = per_fold, fold_id = folds, mode = mode, K = K, n = n,
       ref = list(x = x_ref, wp = wp_ref),
       tangent = tangent, lambda_z = lambda_z, sigma_A = sigma_A, nu_A = nu_A,
       N0 = N0, null_name = null$name, call = match.call())
}

## =============================================================
## Test statistic from a per-fold record list (both modes, quadrature
## calibration). No simulation term.
##
## The 2x2: rows raw / orth, columns X (held-out data) / Q (fitted-null
## quadrature). BOTH rows use the same numerator S = sum_k S_k built from
## the raw witness; the rows differ in the variance only:
##   raw   sum_i-or-int  phi^2                    (valid only if g = 0)
##   orth  sum_i-or-int (phi - b^T u)^2 + v_aux   (influence / sandwich)
## This is "correction in the variance" (Theorem B.1, general
## studentization). Module 3's "correction in the statistic" -- replace
## phi by phi~ in the numerator too, as cvgof_test() does for the KDE
## pipeline -- is equivalent to it at second order in cond mode and in
## unconstrained pois mode (the test-fold score at the training MLE sums
## to zero over folds), but NOT under an auxiliary constraint: there
## phi~ retains the plug-in direction g~ = Sigma_A^{-1} I_k^{-1} g, and
## S_orth differs from S at first order by subtracting the unabsorbed
## count fluctuation, i.e. the channel the constraint makes detectable.
## S_orth is therefore returned as a diagnostic (orth_num) with its own
## Z valid in the two unconstrained cases, and the headline is
## Z = S / sqrt(V_orth_Q + v_aux).
## =============================================================
cvgof_test_quad <- function(cv_out, alpha = 0.05) {
  pf <- cv_out$per_fold
  K  <- length(pf); n <- cv_out$n
  d  <- ncol(pf[[1]]$u_grid)
  centered <- isTRUE(pf[[1]]$centered)
  SigAinv  <- pf[[1]]$SigAinv

  S_raw <- S_orth <- VQ_raw <- VQ_orth <- ellr <- m_k <- numeric(K)
  g <- b <- matrix(NA_real_, K, d)
  c_aux <- numeric(d)
  phi_all <- phit_all <- numeric(n)
  nu_hat  <- matrix(NA_real_, K, length(pf[[1]]$nu),
                    dimnames = list(NULL, names(pf[[1]]$nu)))

  for (k in seq_len(K)) {
    f <- pf[[k]]
    w_te <- f$omega_te; w_tr <- f$omega_tr
    ph_g <- f$phi_grid; ph_x <- f$phi_x
    u_g  <- f$u_grid;   u_x  <- f$u_x
    stopifnot(length(ph_x) == length(f$idx_ev), length(ph_g) == length(w_te))

    ## held-out ELLR and its fitted-null mean (pois) / plain sum and
    ## N_k E phi (cond); the difference is the centered statistic
    if (centered) {
      ellr[k] <- sum(ph_x);  m_k[k] <- sum(w_te * ph_g)
    } else {
      ellr[k] <- sum(ph_x) - sum(w_te * (exp(ph_g) - 1))
      m_k[k]  <- -sum(w_te * (exp(ph_g) - 1 - ph_g))
    }
    S_raw[k] <- sum(ph_x) - sum(w_te * ph_g)

    ## plug-in direction and (K-1)-scaled orthogonalization coefficient
    g[k, ]  <- colSums(w_te * ph_g * u_g)
    I_k     <- crossprod(u_g, u_g * w_tr) + SigAinv
    Iinv_g  <- solve(I_k, g[k, ])
    b[k, ]  <- (K - 1) * Iinv_g
    c_aux   <- c_aux + Iinv_g

    pht_g <- ph_g - as.vector(u_g %*% b[k, ])
    pht_x <- ph_x - as.vector(u_x %*% b[k, ])
    S_orth[k] <- sum(pht_x) - sum(w_te * pht_g)

    vq <- function(p) sum(w_te * p^2) - if (centered) sum(w_te * p)^2 / f$N_k else 0
    VQ_raw[k]  <- vq(ph_g)
    VQ_orth[k] <- vq(pht_g)

    phi_all[f$idx_ev]  <- ph_x
    phit_all[f$idx_ev] <- pht_x
    nu_hat[k, ] <- f$nu
  }

  vx <- function(p) sum(p^2) - if (centered) sum(p)^2 / n else 0
  V_tab <- rbind(X = c(raw = vx(phi_all),  orth = vx(phit_all)),
                 Q = c(raw = sum(VQ_raw),  orth = sum(VQ_orth)))
  v_aux <- as.numeric(crossprod(c_aux, SigAinv %*% c_aux))

  ## --- cross-fold covariance: the fitting randomness -------------------
  ## sum_k V_k is E Var(S_k | D_-k). Under a correctly specified null the
  ## witness is pure estimation noise and the fold statistics are
  ## correlated through their shared training data: to first order
  ##   Cov(S_j, S_k) = E_j E_k tr( D_k H_k^-1 M_kj D_j H_j^-1 M_jk ),
  ## H_k the fold-k GLM Hessian on its active set, M_kj the cross-Gram of
  ## the two folds' active atoms under the fitted null, D_k the tangent
  ## damping diag(1 - rho), E_k the expected test count. For shared
  ## orthonormal unpenalized atoms this is E_j E_k p / expo_tr^2 and the
  ## sum over j != k equals sum_k V_k: the factor-2 doubling of the
  ## conditional variance. Under H1 it stays at the noise scale.
  V_cross <- 0; p_act <- NA_real_
  has_ref <- !is.null(pf[[1]]$A_act_ref) && !is.null(cv_out$ref)
  if (has_ref) {
    wp_ref <- cv_out$ref$wp
    Hinv <- lapply(pf, function(f) {
      H <- crossprod(f$A_act_ref, f$A_act_ref * f$mu_ref)
      f$damp * solve(H)                             # D_k H_k^-1 (row scaling)
    })
    p_act <- mean(vapply(pf, function(f) ncol(f$A_act_ref), integer(1)))
    for (j in seq_len(K - 1L)) for (k in (j + 1L):K) {
      M_kj <- crossprod(pf[[k]]$A_act_ref, pf[[j]]$A_act_ref * wp_ref)   # p_k x p_j
      C_jk <- pf[[j]]$expo_te * pf[[k]]$expo_te *
              sum((Hinv[[k]] %*% M_kj) * t(Hinv[[j]] %*% t(M_kj)))       # tr(D_k Hk^-1 M_kj D_j Hj^-1 M_jk)
      V_cross <- V_cross + 2 * C_jk                                     # ordered pairs
    }
  }

  stat <- function(S_k, V_x, V_q, v_extra) {
    S <- sum(S_k)
    list(S = S, S_k = S_k, V_x = V_x, V_q = V_q, v_extra = v_extra,
         Z_X = S / sqrt(V_x + v_extra), Z_Q = S / sqrt(V_q + v_extra),
         se_X = sqrt(V_x + v_extra), se_Q = sqrt(V_q + v_extra))
  }
  raw      <- stat(S_raw, V_tab["X", "raw"],  V_tab["Q", "raw"],  0)
  orth     <- stat(S_raw, V_tab["X", "orth"], V_tab["Q", "orth"], v_aux)  # same numerator
  orth_num <- stat(S_orth, V_tab["X", "orth"], V_tab["Q", "orth"], 0)     # diagnostic
  fit      <- stat(S_raw, V_tab["X", "orth"], V_tab["Q", "orth"], v_aux + V_cross)  # + fitting randomness
  head     <- if (has_ref) fit else orth

  structure(list(
    Z          = head$Z_Q,
    p_value    = stats::pnorm(head$Z_Q, lower.tail = FALSE),
    reject     = stats::pnorm(head$Z_Q, lower.tail = FALSE) < alpha,
    fit = fit, V_cross = V_cross, p_act = p_act,
    orth = orth, raw = raw, orth_num = orth_num,
    ellr = ellr, m_k = m_k,             # ellr - m_k == raw$S_k
    g_hat = g, b = b, v_aux = v_aux, nu_hat = nu_hat,
    V_tab = V_tab,
    sandwich_X = V_tab["X", "raw"] / V_tab["X", "orth"],
    sandwich_Q = V_tab["Q", "raw"] / V_tab["Q", "orth"],
    ratio_raw  = V_tab["X", "raw"]  / V_tab["Q", "raw"],
    ratio_orth = V_tab["X", "orth"] / V_tab["Q", "orth"],
    phi = phi_all, phi_orth = phit_all,
    mode = cv_out$mode, n = n, K = K, alpha = alpha
  ), class = "cvgof_test")
}

## ---------------------------------------------------------------
## Fixed-witness check of cvgof_test_quad(): a KNOWN phi (no learning),
## Poisson data, per-fold constrained null fits, B toys. With phi fixed
## the statistic's variance is exactly what the studentizer targets, so
##   var(S_orth) / mean(V_orth_Q + v_aux) ~ 1
## is a direct test of the (K-1) factor and of v_aux. Also reported:
## the same ratio for the doc's un-scaled formula and for the raw row.
## ---------------------------------------------------------------
check_test_quad_fixed_witness <- function(B = 2000L, K = 10L,
                                          phi_fun = function(x) 0.1 + 0.05 * x + 0.3 * exp(-(x - 3)^2),
                                          sigma_A = c(0.05, 0.05), nu_star = c(0, 0),
                                          N0 = N0_DEFAULT, n_grid = 4001L, seed = 1,
                                          balanced = TRUE) {
  set.seed(seed)
  nul <- null_exponential()
  res <- t(vapply(seq_len(B), function(bb) {
    d  <- gen_data_pois(signal = FALSE, nu_star = nu_star, sigma_A = sigma_A, N0 = N0)
    f  <- make_folds(d$N, K, balanced = balanced)
    pf <- vector("list", K); pf_doc <- pf
    for (k in seq_len(K)) {
      idx <- which(f == k); x_tr <- d$x[which(f != k)]
      fr  <- fold_null_frame("pois", x_tr, N_k = length(idx), K = K, null = nul,
                             nu_A = d$nu_A, sigma_A = sigma_A, N0 = N0)
      qg  <- make_quad_grid(nul, fr$theta, n_grid = n_grid)
      pf[[k]] <- make_fold_record(fr, qg$x_grid, qg$wp, idx, d$x[idx],
                                  phi_fun(d$x[idx]), phi_fun(qg$x_grid))
    }
    r <- cvgof_test_quad(list(per_fold = pf, n = d$N, mode = "pois"))
    ## doc's version: b/(K-1)
    S_doc <- 0; V_doc <- 0
    for (k in seq_len(K)) { p <- pf[[k]]; bd <- r$b[k, ] / (K - 1)
      S_doc <- S_doc + sum(p$phi_x - p$u_x %*% bd) - sum(p$omega_te * (p$phi_grid - p$u_grid %*% bd))
      V_doc <- V_doc + sum(p$omega_te * (p$phi_grid - p$u_grid %*% bd)^2) }
    c(S = r$raw$S, S_orth = r$orth_num$S, V_orth_Q = r$orth$V_q, V_orth_X = r$orth$V_x,
      v_aux = r$v_aux, V_raw_Q = r$raw$V_q, S_doc = S_doc, V_doc = V_doc,
      Z = r$Z, Z_raw_Q = r$raw$Z_Q, Z_orthnum_Q = r$orth_num$Z_Q)
  }, numeric(11)))
  res <- as.data.frame(res)
  list(
    ratio_sand_Q_with_aux = var(res$S) / mean(res$V_orth_Q + res$v_aux),   # ~ 1 (the headline)
    ratio_sand_Q_no_aux   = var(res$S) / mean(res$V_orth_Q),               # > 1 iff aux matters
    ratio_sand_X_with_aux = var(res$S) / mean(res$V_orth_X + res$v_aux),   # ~ 1
    ratio_orthnum_Q       = var(res$S_orth) / mean(res$V_orth_Q),          # ~ 1 only if unconstrained
    ratio_doc_formula     = var(res$S_doc)  / mean(res$V_doc),             # != 1 unless Sigma_A^{-1} = 0... and not even then
    ratio_raw_Q           = var(res$S) / mean(res$V_raw_Q),                # < 1 (conservative)
    aux_share             = mean(res$v_aux) / mean(res$V_orth_Q + res$v_aux),
    S_orth_vs_S           = c(cor = cor(res$S, res$S_orth), sd_ratio = sd(res$S_orth) / sd(res$S)),
    Z_mean_sd             = c(mean = mean(res$Z), sd = sd(res$Z)),         # ~ (0, 1)
    Z_raw_sd              = sd(res$Z_raw_Q),
    Z_orthnum_sd          = sd(res$Z_orthnum_Q),
    B = B, K = K, sigma_A = sigma_A, results = res)
}

## =============================================================
## Toy loop: tilt pipeline, cond or pois mode
## =============================================================
## Usage:
##   source("utils.R")
##   ## conditional (iid) mode, NP2 alternative, universal threshold
##   r1 <- run_tilt_experiment(mode = "cond", signal = TRUE, sig = "NP2")
##   ## Poisson mode, NPLM sigma = 0.1, null study
##   r0 <- run_tilt_experiment(mode = "pois", signal = FALSE,
##                             sigma_A = c(0.1, 0.1))
## Auto tag: <mode>_<alt_SIG|null>_<n|N0>..._lam<U|value>_K<K>_s<seed>,
## pois tags carry sA<sigma_n>_<sigma_s> (Inf -> "inf") and the true
## nuisance nu* when nonzero. As in run_cvgof_experiment, n_toys is
## left out of the tag and the null DGP does not depend on sig.
## In pois mode a null run IS tied to (N0, sigma_A, nu*), not to sig.
## Columns: Z_raw_{X,Q}, Z_orth_{X,Q} (headline Z_orth_Q; orth = raw
## numerator + sandwich variance), Z_orthnum_Q (diagnostic: phi~ in the
## numerator), S (numerator), V's, v_aux, sandwich ratios, g_hat and b
## per direction, n_active bumps, tangent coefficients, nu_hat.
run_tilt_experiment <- function(mode      = c("cond", "pois"),
                                signal    = TRUE,
                                sig       = SIG,
                                n         = NULL,             # cond only
                                N0        = N0_DEFAULT,       # pois only
                                nu_star   = c(0, 0),          # pois only
                                sigma_A   = c(0.1, 0.1),      # pois only
                                N_sig     = NULL,             # pois only (fixed signal count)
                                lambda_z  = NULL,             # NULL = universal
                                tangent   = NULL,             # NULL = (mode == "pois")
                                dict_args = list(),
                                fit_args  = list(),
                                n_toys    = 400,
                                K         = K_FOLDS,
                                seed      = 1001,
                                cores     = 1,
                                chunk     = 25,
                                tag       = NULL,
                                save_dir  = ".",
                                verbose   = TRUE,
                                overwrite = FALSE,
                                bump_damping = c("none", "soft")) {
  mode <- match.arg(mode); bump_damping <- match.arg(bump_damping)
  stopifnot(is.character(sig), length(sig) == 1L, sig %in% names(SIGNALS),
            n_toys >= 1, chunk >= 1, length(sigma_A) == 2L, length(nu_star) == 2L)
  sg <- SIGNALS[[sig]]
  if (is.null(tangent)) tangent <- (mode == "pois")
  if (mode == "cond" && is.null(n)) {
    if (!signal) stop("run_tilt_experiment: n must be given explicitly when ",
                      "signal = FALSE in cond mode (null DGP does not depend on sig)")
    n <- sg$n_def
  }

  fmt_s <- function(v) if (is.infinite(v)) "inf" else .fmt_num(v)
  if (is.null(tag))
    tag <- sprintf("%s_%s%s_%s%s_lam%s_K%d_s%d",
                   mode,
                   if (signal) "alt" else "null",
                   if (signal) paste0("_", sig) else "",
                   if (mode == "cond") sprintf("n%d", n)
                   else sprintf("N%d_sA%s_%s", N0, fmt_s(sigma_A[1]), fmt_s(sigma_A[2])),
                   if (mode == "pois" && any(nu_star != 0))
                     sprintf("_nu%s_%s", .fmt_num(nu_star[1]), .fmt_num(nu_star[2])) else "",
                   if (is.null(lambda_z)) "U" else .fmt_num(lambda_z),
                   K, seed)
  ## tags: runs made before the "none" default have no suffix and used "soft";
  ## every run since carries its rule explicitly (_dmpnone / _dmpsoft)
  if (!grepl("_dmp", tag)) tag <- paste0(tag, "_dmp", bump_damping)

  t_start <- Sys.time()
  if (cores > 1) suppressPackageStartupMessages(library(parallel))
  set.seed(seed)
  toy_seeds <- sample.int(.Machine$integer.max, n_toys)

  run_one <- function(i) {
    set.seed(toy_seeds[i])
    if (mode == "cond") {
      d  <- gen_data(n = n, sig = sig, signal = signal)
      cv <- cvgof_fit_tilt(d$x, K = K, mode = "cond", lambda_z = lambda_z,
                           tangent = tangent, dict_args = dict_args, fit_args = fit_args,
                           bump_damping = bump_damping)
      nu_A <- c(NA_real_, NA_real_)
    } else {
      d  <- gen_data_pois(sig = sig, nu_star = nu_star, sigma_A = sigma_A,
                          N0 = N0, signal = signal, N_sig = N_sig)
      cv <- cvgof_fit_tilt(d$x, K = K, mode = "pois", lambda_z = lambda_z,
                           nu_A = d$nu_A, sigma_A = sigma_A, N0 = N0,
                           tangent = tangent, dict_args = dict_args, fit_args = fit_args,
                           bump_damping = bump_damping)
      nu_A <- d$nu_A
    }
    r  <- cvgof_test_quad(cv)
    pf <- cv$per_fold
    n_act <- vapply(pf, function(f) length(f$active_bumps), integer(1))
    btan  <- if (tangent) vapply(pf, function(f) f$beta_tan[1], numeric(1)) else NA_real_
    d_u   <- ncol(r$g_hat)

    row <- data.frame(
      toy = i, seed = toy_seeds[i], N = length(d$x), n_sig = d$n_sig,
      Z_raw_X = r$raw$Z_X,   Z_orth_X = r$orth$Z_X,
      Z_raw_Q = r$raw$Z_Q,   Z_orth_Q = r$orth$Z_Q,
      Z_fit_X = r$fit$Z_X,   Z_fit_Q = r$fit$Z_Q,
      V_cross = r$V_cross,   p_act = r$p_act,
      Z_orthnum_Q = r$orth_num$Z_Q,
      S = r$raw$S, S_orth = r$orth_num$S,
      ellr = sum(r$ellr), m_null = sum(r$m_k),
      V_raw_X = r$V_tab["X", "raw"],  V_orth_X = r$V_tab["X", "orth"],
      V_raw_Q = r$V_tab["Q", "raw"],  V_orth_Q = r$V_tab["Q", "orth"],
      v_aux = r$v_aux,
      sandwich_X = r$sandwich_X, sandwich_Q = r$sandwich_Q,
      n_active_mean = mean(n_act), n_active_max = max(n_act),
      beta_tan_mean = mean(btan),
      lambda_z = pf[[1]]$lambda_z, J = pf[[1]]$J,
      kkt_max = max(vapply(pf, function(f) f$kkt, numeric(1)))
    )
    for (j in seq_len(d_u)) {
      row[[paste0("g", j, "_mean")]] <- mean(r$g_hat[, j])
      row[[paste0("g", j, "_sd")]]   <- stats::sd(r$g_hat[, j])
      row[[paste0("b", j, "_mean")]] <- mean(r$b[, j])
    }
    row$nu_n_hat <- mean(r$nu_hat[, "nu_n"]); row$nu_s_hat <- mean(r$nu_hat[, "nu_s"])
    row$nu_A_n <- nu_A[1]; row$nu_A_s <- nu_A[2]
    row
  }

  meta <- list(pipeline = "tilt", mode = mode, signal = signal,
               sig = if (signal) sig else NA_character_,
               n = if (mode == "cond") n else NA_integer_,
               N0 = if (mode == "pois") N0 else NA_real_,
               nu_star = if (mode == "pois") nu_star else NA,
               sigma_A = if (mode == "pois") sigma_A else NA,
               N_sig = N_sig, lambda_z = lambda_z, tangent = tangent, bump_damping = bump_damping,
               dict_args = dict_args, fit_args = fit_args,
               n_toys = n_toys, K = K, seed = seed, cores = cores, tag = tag,
               lambda = if (signal) sg$lambda else NA_real_,
               started = t_start, sessionInfo = utils::sessionInfo())

  if (verbose) {
    cat(sprintf("\ntag=%s | mode=%s signal=%s sig=%s K=%d n_toys=%d seed=%d\n",
                tag, mode, signal, if (signal) sig else "-", K, n_toys, seed))
    if (mode == "cond") cat(sprintf("n=%d ", n))
    else cat(sprintf("N0=%d sigma_A=(%s, %s) nu*=(%g, %g) ", N0,
                     fmt_s(sigma_A[1]), fmt_s(sigma_A[2]), nu_star[1], nu_star[2]))
    cat(sprintf("lambda_z=%s tangent=%s cores=%d\n",
                if (is.null(lambda_z)) "universal" else format(lambda_z), tangent, cores))
    flush.console()
  }

  out <- .toy_loop(run_one, n_toys, meta, tag, save_dir, chunk, cores,
                   verbose, overwrite, t_start)
  if (verbose) print(out)
  invisible(out)
}

## ---------------------------------------------------------------
## Penalty protocol: choose lambda_z so that, under H0, the mean
## fraction of bump coefficients that are nonzero is <= 5%.
##   B = 250 simulated null TRAINING folds (same n_tr, K, fitted null
##   and dictionary as the pipeline); full lasso path on each; count
##   active bumps at every lambda; lambda_z = smallest grid value with
##   mean(#active)/J <= 0.05. Fix it before the null and power studies.
## ---------------------------------------------------------------
choose_lambda <- function(mode = c("cond", "pois"), B = 250, alpha_active = 0.05,
                          n = 2010, K = 10,                        # cond
                          sigma_A = c(0.1, 0.1), N0 = 2000,        # pois
                          lambda_grid = exp(seq(log(6), log(0.5), length.out = 41)),
                          seed = 7001) {
  mode <- match.arg(mode); set.seed(seed)
  nul <- null_exponential(); tangent <- (mode == "pois")
  lambda_grid <- sort(lambda_grid, decreasing = TRUE)

  active <- t(replicate(B, {
    if (mode == "cond") {                          # one training fold, iid
      x_tr <- rexp(round(n * (K - 1) / K)); th <- 1 / mean(x_tr); expo <- length(x_tr)
    } else {                                       # one training fold, Poisson + constraint
      d  <- gen_data_pois(signal = FALSE, sigma_A = sigma_A, N0 = N0)
      f  <- sample.int(K, d$N, replace = TRUE); x_tr <- d$x[which(f != 1)]
      fr <- fold_null_frame("pois", x_tr, N_k = sum(f == 1), K = K, null = nul,
                            nu_A = d$nu_A, sigma_A = sigma_A, N0 = N0)
      th <- fr$theta; expo <- fr$expo_tr
    }
    dict <- make_dict(nul, th, tangent = tangent)
    fit  <- glmnet::glmnet(dict$A_grid, interp_counts(dict, x_tr), family = "poisson",
                           offset = log(expo * dict$wp), penalty.factor = dict$pen_factor,
                           lambda = lambda_to_glmnet(lambda_grid, expo, nrow(dict$A_grid)),
                           standardize = FALSE, intercept = TRUE)
    colSums(as.matrix(fit$beta)[dict$idx_bump, ] != 0)   # active bumps at each lambda
  }))
  J    <- length(make_dict(nul, 1, tangent = tangent)$idx_bump)   # 70
  frac <- colMeans(active) / J
  i    <- max(which(frac <= alpha_active))
  list(lambda_z = lambda_grid[i], mean_active = colMeans(active)[i],
       curve = data.frame(lambda_z = lambda_grid, frac_active = frac))
}

if (FALSE) {   # penalty protocol, run once by hand; results: cond 1.351, pois 1.270
  lam_cond <- choose_lambda("cond")
  lam_pois <- choose_lambda("pois", sigma_A = c(0.1, 0.1), seed = 7002)
}
