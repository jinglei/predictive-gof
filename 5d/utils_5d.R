## =============================================================
## utils_5d.R -- cvgof on the NPLM 5-D dilepton benchmark
## d'Agnolo et al., EPJC 82:275 (2022), Sect. 4; data: Zenodo 4442665
## =============================================================
## Part 1: data layer
##   1.1 configuration
##   1.2 HDF5 I/O with RDS cache
##   1.3 selection cuts, pool trimming, momentum-scale nuisance
##   1.4 feature maps
##   1.5 pools: delta pool / toy+reference pool / signal pools
##   1.6 toy generation (D at nu*, reference R at nu = 0, aligned folds)
##   1.7 delta-network training samples (NPLM Eqs. 15-17) and export
##   1.8 data checks
##
## Conventions
##   * GeV. Events are canonicalized on load: pt1 >= pt2 (leading first),
##     delta_phi = phi1 - phi2 wrapped to [0, 2 pi), mll as stored.
##   * nu = c(nu_n, nu_s) in absolute units, as in utils.R. nu_n is the
##     log normalization shift (sigma_n = 0.025); nu_s the barrel log
##     momentum-scale shift (sigma_s = 5e-4 muon-like, 3e-3 electron-like).
##     Endcap leptons (|eta| >= 2.1) shift by ratio_e * nu_s (NPLM: 3).
##   * N0 = N(R0): expected SM count after cuts at nu = 0. NS: expected
##     signal count after cuts at nu = 0. e^{nu_n} scales both.
##   * Reference weights w_R = N0 / N_R, so sum_R w_R = N0 (NPLM Eq. 22).
##     The reference is always central (nu = 0).
##   * Pools are pre-trimmed to a loose acceptance (cuts relaxed by a
##     factor e^{trim_log}) so the nuisance can move events across the
##     cuts. For each nu_s the pool is transformed once (cached) and the
##     events passing the cuts are recorded. A toy then draws
##       N ~ Poisson(N_data e^{nu_n} pass(nu_s) / pass(0))
##     exactly, and N passing events without replacement. (Drawing a
##     Poisson number of raw events without replacement and cutting
##     afterwards gives hypergeometric, under-dispersed counts: var/mean
##     ~ (1 - p)(1 - f) + p with f the raw sampling fraction.)
##   * The delta pool is disjoint from the toy/reference pool. Toys and
##     references are resampled from the toy pool as in NPLM Sect. 4.1
##     (each toy uses a small fraction of it; references overlap across
##     toys but are disjoint from their own toy's data).
##   * Two representations per event. The null/nuisance model is NPLM's,
##     in 5-D: delta_1, delta_2 are trained on the raw five features
##     ("raw5") exactly as in NPLM Sect. 4.2 and evaluated on 5-D events.
##     Only the witness is reduced: f depends on x through the boost-
##     invariant, symmetry-folded T = (pt1, pt2, cosh(deta), cos(dphi))
##     ("dict4"), i.e.
##       n(x|H) = n(x|R0) exp(nu_n + nu_s d1(x) + nu_s^2 d2(x)/2) e^{f(T(x))}.
##     "nn4" is a bijection of T, kept for ablations.
## =============================================================

## -------------------------------------------------------------
## 1.1 Configuration
## -------------------------------------------------------------
CFG5 <- list(
  data_dir   = "data",    # contains DiLepton_SM/, Zprime300/, ...
  cache_dir  = "cache",
  vars       = c("pt1", "pt2", "eta1", "eta2", "delta_phi", "mll"),
  cuts       = list(pt_min = 20, pt_max = 1000, mll_min = 100, eta_max = Inf),
  trim_log   = 0.065,                # pools keep events within e^{+-0.065} of the cuts;
                                     # must be >= ratio_e * max|nu_s| used anywhere
                                     # (3 * 0.02 for the Fig.-13 check grid)
  N0         = 8700,                 # NPLM Sect. 4 (muon/electron, mll > 100)
  NR_ratio   = 5,                    # N_R = 5 N0 (NPLM Sect. 4.1)
  K          = 10L,
  sigma_A    = c(nu_n = 0.025, nu_s = 3e-3),   # electron-like
  ratio_e    = 3,
  eta_edge   = 2.1,
  nu_delta   = c(-1.5e-2, -1.5e-3, 1.5e-3, 1.5e-2),   # NPLM Sect. 4.2
  frac_delta = 0.25,
  frac_ref   = 0,                    # > 0: separate reference pool (not recommended for toys, see make_pools_5d)
  resample   = "iid",                # toy draws: "iid" (with replacement, D and R independent) or "exclude"
  seed_pool  = 20260924L
)

## Zenodo directory names; resolve_sample_dir_5d() also accepts the short
## form ("Zprime300" -> "DiLepton_Zprime300").
`%||%` <- function(a, b) if (is.null(a)) b else a

SAMPLES_5D <- c("DiLepton_SM", "DiLepton_Zprime200", "DiLepton_Zprime300",
                "DiLepton_Zprime600", "DiLepton_EFT06", "DiLepton_EFT06_2",
                "DiLepton_EFT06_5")

## Benchmarks (NPLM Sect. 4.4). sigma_A in absolute units; nu_star = +1
## sigma_s as in NPLM's power runs. The EFT sample-to-c_W mapping is not
## yet established (see the Table-3 fit), so no EFT entry is hard-coded.
BENCH_5D <- list(
  zp300_mu = list(data = "SM", signal = "Zprime300", NS = 120,
                  sigma_A = c(nu_n = 0.025, nu_s = 5e-4), nu_star = c(0, 5e-4)),
  zp300_e  = list(data = "SM", signal = "Zprime300", NS = 120,
                  sigma_A = c(nu_n = 0.025, nu_s = 3e-3), nu_star = c(0, 3e-3))
)

## -------------------------------------------------------------
## 1.2 HDF5 I/O with RDS cache
## -------------------------------------------------------------
.h5_backend <- function() {
  if (requireNamespace("rhdf5", quietly = TRUE)) return("rhdf5")
  if (requireNamespace("hdf5r", quietly = TRUE)) return("hdf5r")
  stop("need package 'rhdf5' (Bioconductor) or 'hdf5r' (CRAN)")
}

.h5_read_file <- function(path, vars, backend = .h5_backend()) {
  if (backend == "rhdf5") {
    out <- lapply(vars, function(v) as.numeric(rhdf5::h5read(path, v)))
    rhdf5::h5closeAll()
  } else {
    f <- hdf5r::H5File$new(path, mode = "r")
    on.exit(f$close_all(), add = TRUE)
    out <- lapply(vars, function(v) as.numeric(f[[v]][]))
  }
  names(out) <- vars
  if (length(unique(lengths(out))) != 1L)
    stop(".h5_read_file: unequal column lengths in ", basename(path))
  as.data.frame(out)
}

## HDF5 data files in a directory, excluding macOS AppleDouble "._*" files.
.h5_files <- function(dir, pattern = "\\.(h5|hdf5)$") {
  fs <- sort(list.files(path.expand(dir), pattern = pattern, full.names = TRUE))
  fs[!startsWith(basename(fs), "._")]
}

## Dataset names at the root of one file (character(0) plus an attribute
## "error" if the file cannot be opened).
.h5_names <- function(path, backend = .h5_backend()) {
  tryCatch({
    if (backend == "rhdf5") {
      ls <- rhdf5::h5ls(path); rhdf5::h5closeAll()
      ls$name[ls$otype == "H5I_DATASET" & ls$group == "/"]
    } else {
      f <- hdf5r::H5File$new(path, mode = "r"); on.exit(f$close_all(), add = TRUE)
      f$ls()$name
    }
  }, error = function(e) {
    L <- trimws(strsplit(conditionMessage(e), "\n")[[1]])
    m <- grep("^minor:", L, value = TRUE)
    structure(character(0), error = if (length(m)) m[length(m)] else substr(L[1], 1, 100))
  })
}

## Validate every file before reading. `empty` marks a valid HDF5 file
## with no datasets (e.g. DiLepton_SM59.h5, 800 bytes in the original
## Zenodo tarball); such files are skipped by read_h5_dir().
check_h5_dir_5d <- function(dir, vars = CFG5$vars, pattern = "\\.(h5|hdf5)$",
                            backend = .h5_backend()) {
  fs <- .h5_files(dir, pattern)
  if (!length(fs)) stop("no HDF5 files in ", dir)
  info  <- lapply(fs, .h5_names, backend = backend)
  empty <- vapply(info, function(nm) !length(nm) && is.null(attr(nm, "error")), logical(1))
  ok    <- vapply(info, function(nm) all(vars %in% nm), logical(1))
  data.frame(file = basename(fs), size_MB = round(file.size(fs) / 2^20, 3),
             ok = ok, empty = empty,
             datasets = vapply(info, function(nm) {
               e <- attr(nm, "error"); if (!is.null(e)) paste("<unreadable>", e)
               else paste(nm, collapse = ",") }, character(1)),
             stringsAsFactors = FALSE)
}

read_h5_dir <- function(dir, vars = CFG5$vars, pattern = "\\.(h5|hdf5)$") {
  fs <- .h5_files(dir, pattern)
  if (!length(fs)) stop("read_h5_dir: no HDF5 files in ", dir)
  be  <- .h5_backend()
  chk <- check_h5_dir_5d(dir, vars, pattern, be)
  bad <- !chk$ok & !chk$empty
  if (any(bad)) {
    print(chk[bad, ], row.names = FALSE)
    stop("read_h5_dir: ", sum(bad), " of ", nrow(chk), " files in ", basename(dir),
         " are unreadable or lack required datasets (listed above); median file size ",
         stats::median(chk$size_MB), " MB")
  }
  if (any(chk$empty))
    message("read_h5_dir: skipping ", sum(chk$empty), " empty file(s) in ", basename(dir), ": ",
            paste(chk$file[chk$empty], collapse = ", "))
  use <- fs[chk$ok]
  X  <- do.call(rbind, lapply(use, .h5_read_file, vars = vars, backend = be))
  attr(X, "files") <- basename(use)
  attr(X, "skipped") <- chk$file[chk$empty]
  X
}

## Canonical event layout: finite rows, delta_phi in [0, 2 pi), pt1 >= pt2.
## A swap exchanges (pt, eta) and sends delta_phi -> 2 pi - delta_phi.
## Row order is preserved (callers rely on it to keep labels aligned).
.canon_events <- function(X) {
  X$delta_phi <- X$delta_phi %% (2 * pi)
  sw <- X$pt2 > X$pt1
  if (any(sw)) {
    p <- X$pt1[sw]; X$pt1[sw] <- X$pt2[sw]; X$pt2[sw] <- p
    e <- X$eta1[sw]; X$eta1[sw] <- X$eta2[sw]; X$eta2[sw] <- e
    X$delta_phi[sw] <- (2 * pi - X$delta_phi[sw]) %% (2 * pi)
  }
  X
}

## Sample directory under data_dir: exact name, else the unique
## subdirectory whose name ends in "_<name>" (e.g. Zprime300 ->
## DiLepton_Zprime300). Errors list what is there.
resolve_sample_dir_5d <- function(name, data_dir = CFG5$data_dir) {
  root <- path.expand(data_dir)
  if (!dir.exists(root)) stop("data_dir does not exist: ", root)
  if (dir.exists(file.path(root, name))) return(file.path(root, name))
  subs <- list.dirs(root, full.names = FALSE, recursive = FALSE)
  hit  <- subs[endsWith(subs, paste0("_", name))]
  if (length(hit) == 1L) return(file.path(root, hit))
  stop("no unique sample directory for '", name, "' in ", root,
       "; subdirectories: ", paste(subs, collapse = ", "))
}

load_sample_5d <- function(name, data_dir = CFG5$data_dir,
                           cache_dir = CFG5$cache_dir, refresh = FALSE) {
  dir   <- resolve_sample_dir_5d(name, data_dir)
  cache <- file.path(path.expand(cache_dir), paste0(basename(dir), ".rds"))
  if (!refresh && file.exists(cache)) return(readRDS(cache))
  X  <- read_h5_dir(dir)
  n0 <- nrow(X); files <- attr(X, "files"); skipped <- attr(X, "skipped")
  ok <- stats::complete.cases(X) & apply(is.finite(as.matrix(X)), 1L, all)
  X  <- .canon_events(X[ok, , drop = FALSE])
  rownames(X) <- NULL
  attr(X, "name") <- basename(dir); attr(X, "n_dropped") <- n0 - nrow(X)
  attr(X, "files") <- files; attr(X, "skipped") <- skipped
  dir.create(dirname(cache), recursive = TRUE, showWarnings = FALSE)
  saveRDS(X, cache)
  X
}

## -------------------------------------------------------------
## 1.3 Cuts, pool trimming, momentum-scale nuisance
## -------------------------------------------------------------
pass_cuts_5d <- function(X, cuts = CFG5$cuts) {
  X$pt2 > cuts$pt_min & X$pt1 < cuts$pt_max & X$mll > cuts$mll_min &
    abs(X$eta1) < cuts$eta_max & abs(X$eta2) < cuts$eta_max
}

## Loose acceptance: cuts relaxed by e^{trim_log}. Valid for any scale
## shift with ratio_e * |nu_s| <= trim_log (pt and mll move by at most
## that factor in log).
trim_5d <- function(X, cuts = CFG5$cuts, trim_log = CFG5$trim_log) {
  t <- exp(trim_log)
  X$pt2 > cuts$pt_min / t & X$pt1 < cuts$pt_max * t & X$mll > cuts$mll_min / t &
    abs(X$eta1) < cuts$eta_max & abs(X$eta2) < cuts$eta_max
}

## NPLM Sect. 4: pT^(b,e) -> exp(nu_s * sigma^(b,e)/sigma^(b)) pT^(b,e),
## barrel/endcap fully correlated through one nu_s. Massless leptons at
## fixed angles give mll -> mll * sqrt(f1 f2). Re-canonicalizes, since
## the shift can reverse the pT ordering.
scale_nuisance_5d <- function(X, nu_s, ratio_e = CFG5$ratio_e,
                              eta_edge = CFG5$eta_edge, trim_log = CFG5$trim_log) {
  if (nu_s == 0) return(X)
  if (!is.null(trim_log) && ratio_e * abs(nu_s) > trim_log)
    stop("scale_nuisance_5d: ratio_e * |nu_s| = ", ratio_e * abs(nu_s),
         " exceeds pool trim_log = ", trim_log)
  f1 <- exp(nu_s * ifelse(abs(X$eta1) >= eta_edge, ratio_e, 1))
  f2 <- exp(nu_s * ifelse(abs(X$eta2) >= eta_edge, ratio_e, 1))
  X$mll <- X$mll * sqrt(f1 * f2)
  X$pt1 <- X$pt1 * f1
  X$pt2 <- X$pt2 * f2
  .canon_events(X)
}

## -------------------------------------------------------------
## 1.4 Feature maps
## -------------------------------------------------------------
##   dict4  (pt1, pt2, cosh(deta), cos(dphi))       dictionary coordinates;
##          even in deta and in dphi -> 2 pi - dphi (exact folds);
##          m^2 = 2 pt1 pt2 (ch_deta - c_dphi) is a cubic in them
##   nn4    (log pt1, log pt2, |deta|, folded dphi in [0, pi])
##          the same 4-D space, better conditioned for networks
##          (cos(dphi) piles up near -1 where most events sit)
##   raw5   NPLM's five features: delta-network inputs and tangents
features_5d <- function(X, space = c("dict4", "nn4", "raw5")) {
  space <- match.arg(space)
  deta  <- X$eta1 - X$eta2
  switch(space,
    dict4 = cbind(pt1 = X$pt1, pt2 = X$pt2,
                  ch_deta = cosh(deta), c_dphi = cos(X$delta_phi)),
    nn4   = cbind(lpt1 = log(X$pt1), lpt2 = log(X$pt2), adeta = abs(deta),
                  dphi_f = pmin(X$delta_phi, 2 * pi - X$delta_phi)),
    raw5  = cbind(pt1 = X$pt1, pt2 = X$pt2, eta1 = X$eta1, eta2 = X$eta2,
                  delta_phi = X$delta_phi))
}

## Invariant mass from dict4 coordinates (diagnostics only; never a
## witness input)
mll_from_dict4 <- function(Z) sqrt(2 * Z[, "pt1"] * Z[, "pt2"] * (Z[, "ch_deta"] - Z[, "c_dphi"]))

## -------------------------------------------------------------
## 1.5 Pools
## -------------------------------------------------------------
.make_pool <- function(X, n_full, cuts, trim_log) {
  keep <- trim_5d(X, cuts, trim_log)
  P <- X[keep, , drop = FALSE]; rownames(P) <- NULL
  acc_trim <- mean(pass_cuts_5d(P, cuts))
  list(ev = P, n_full = n_full, n_trim = nrow(P),
       acc_trim = acc_trim, acc_full = acc_trim * nrow(P) / n_full,
       n_pass = sum(pass_cuts_5d(P, cuts)))
}

## sm       SM sample (load_sample_5d("DiLepton_SM"))
## samples  named list of other full samples (signals, EFT); each becomes
##          its own pool, used whole
## Disjoint SM pools from one permutation: delta (training of the delta
## networks) and toy (population for the toys); optionally ref (frac_ref > 0).
## Toy design (see draw_toy_5d, resample = "iid"): the toy pool's empirical
## distribution IS the population; data and reference are independent draws
## from it, with replacement. Two alternatives were tested and rejected:
##   * one pool, reference excluding the toy's data events ("exclude"):
##     R then depends on D (depleted of the migrating events D drew),
##     biasing nu-hat at nu* != 0;
##   * separate toy and ref pools (frac_ref = 0.5): the two halves differ by
##     Monte Carlo noise, a fixed mismatch common to all toys (observed
##     offset ~1e-3 in nu_s-hat at N0 = 8700).
## The delta pool depends only on (seed, frac_delta), so trained delta-hat
## networks stay valid whatever frac_ref is.
make_pools_5d <- function(sm, samples = list(), frac_delta = CFG5$frac_delta,
                          frac_ref = CFG5$frac_ref,
                          cuts = CFG5$cuts, trim_log = CFG5$trim_log,
                          seed = CFG5$seed_pool) {
  stopifnot(frac_delta > 0, frac_delta < 1, frac_ref >= 0, frac_ref < 1, is.list(samples))
  if (length(samples)) stopifnot(!is.null(names(samples)), all(nzchar(names(samples))))
  set.seed(seed)
  n    <- nrow(sm)
  perm <- sample.int(n)
  nd   <- round(frac_delta * n)
  rest <- perm[-seq_len(nd)]
  nr   <- round(frac_ref * length(rest))
  i_ref <- rest[seq_len(nr)]; i_toy <- if (nr > 0) rest[-seq_len(nr)] else rest
  pools <- list(
    delta = .make_pool(sm[perm[seq_len(nd)], , drop = FALSE], nd, cuts, trim_log),
    toy   = .make_pool(sm[i_toy, , drop = FALSE], length(i_toy), cuts, trim_log),
    ref   = if (nr > 0) .make_pool(sm[i_ref, , drop = FALSE], nr, cuts, trim_log) else NULL,
    sig   = lapply(samples, function(S) .make_pool(S, nrow(S), cuts, trim_log)),
    cuts = cuts, trim_log = trim_log, frac_delta = frac_delta, frac_ref = frac_ref, seed = seed)
  class(pools) <- "pools_5d"
  pools
}

print.pools_5d <- function(x, N0 = CFG5$N0, NR_ratio = CFG5$NR_ratio, B = 400L, ...) {
  row <- function(nm, p) cat(sprintf("  %-12s full %9d  trimmed %8d  pass %8d  acc_full %.4f  acc_trim %.4f\n",
                                     nm, p$n_full, p$n_trim, p$n_pass, p$acc_full, p$acc_trim))
  cat("pools_5d (seed ", x$seed, ", frac_delta ", x$frac_delta, ", frac_ref ", x$frac_ref,
      ", trim_log ", x$trim_log, ")\n", sep = "")
  row("delta", x$delta); row("toy", x$toy); if (!is.null(x$ref)) row("ref", x$ref)
  for (nm in names(x$sig)) row(nm, x$sig[[nm]])
  p_D <- N0 / x$toy$n_pass
  p_R <- NR_ratio * N0 / (if (is.null(x$ref)) x$toy else x$ref)$n_pass
  cat(sprintf("  toy sampling fraction per toy: D %.3f, R %.3f (NPLM: ~0.056)\n", p_D, p_R))
  cat(sprintf("  heuristic B_eff for a quadratic-form statistic (rho ~ p_D^2), B = %d: %.0f\n",
              B, B / (1 + (B - 1) * p_D^2)))
  invisible(x)
}

## -------------------------------------------------------------
## 1.6 Toy generation
## -------------------------------------------------------------
## Data D at nuisance nu_star = c(nu_n, nu_s), reference R at nu = 0.
##   data     "SM" (toy pool) or the name of a full-sample pool in
##            pools$sig (e.g. an EFT sample, which replaces the SM, as
##            in NPLM's EFT benchmark)
##   N_data   expected count of the `data` component after cuts at nu = 0
##            (N0 for SM; for an EFT sample, N0 times its cross-section
##            ratio, which must be supplied)
##   signal   name of an additive signal pool (e.g. "Zprime300") or NULL
##   NS       expected signal count after cuts at nu = 0
##   N_sig    if given, exactly this many signal events after cuts
##            (variance-reduced power, as in utils.R)
##   sigma_A  auxiliary-measurement sds; nu_A ~ N(nu_star, sigma_A),
##            Inf leaves nu_A at nu_star
##   resample "iid" (default): data, signal and reference are independent
##            draws WITH replacement from the pool's empirical distribution,
##            so D is an exact Poisson process and R an exact i.i.d. sample
##            of the same population, independent of D (D and R share
##            events by chance, ~ N * N_R / pool size). "exclude": draws
##            without replacement, R disjoint from D's events (biased;
##            kept for reference)
## Returns the per-toy object consumed by the fitting code:
##   x,  xR    witness coordinates (`space`, default dict4) of D and R
##   xn, xnR   raw five features of D and R, for delta-hat and tangents
##   Rpre      pre-selection reference events (data frame, raw layout):
##             R plus the near-threshold events failing at nu = 0, for
##             the pushforward T_nu; pass0 flags R within Rpre; foldRpre
##             its fold ids (foldR = foldRpre[pass0])
## plus reference weights, aligned balanced fold ids, labels z
## (1 = signal event), and mll for diagnostics.
.make_folds5 <- function(n, K) sample(rep_len(seq_len(K), n))

## Cache of transformed pools: for (pool, nu_s) the shifted events and the
## row indices passing the cuts. Row order equals the pool's, so indices
## refer to the same source events at every nu_s.
.pass_cache5 <- new.env(parent = emptyenv())
clear_cache_5d <- function() rm(list = ls(.pass_cache5), envir = .pass_cache5)

.passing_5d <- function(pools, which, nu_s, ratio_e = CFG5$ratio_e, eta_edge = CFG5$eta_edge) {
  P <- if (which %in% c("toy", "ref", "delta")) pools[[which]] else pools$sig[[which]]
  if (is.null(P)) stop("unknown pool '", which, "'")
  key <- paste(which, P$n_trim, P$n_pass, pools$seed, sprintf("%.10g", nu_s),
               ratio_e, eta_edge, sep = "|")
  hit <- .pass_cache5[[key]]
  if (!is.null(hit)) return(hit)
  X   <- scale_nuisance_5d(P$ev, nu_s, ratio_e, eta_edge, pools$trim_log)
  idx <- which(pass_cuts_5d(X, pools$cuts))
  out <- list(X = X, idx = idx, ratio = length(idx) / P$n_pass)
  assign(key, out, envir = .pass_cache5)
  out
}

draw_toy_5d <- function(pools, N0 = CFG5$N0, data = "SM", N_data = N0,
                        signal = NULL, NS = 0, N_sig = NULL,
                        nu_star = c(0, 0), sigma_A = CFG5$sigma_A,
                        NR_ratio = CFG5$NR_ratio, K = CFG5$K, space = "dict4",
                        resample = CFG5$resample,
                        ratio_e = CFG5$ratio_e, eta_edge = CFG5$eta_edge) {
  stopifnot(inherits(pools, "pools_5d"), length(nu_star) == 2L, length(sigma_A) == 2L)
  nu_n <- nu_star[1]; nu_s <- nu_star[2]
  iid  <- identical(resample, "iid")
  pick <- function(pc, n, what) {
    if (!iid && n > length(pc$idx)) stop("draw_toy_5d: ", what, " pool too small (", n, " > ", length(pc$idx), ")")
    pc$idx[sample.int(length(pc$idx), n, replace = iid)]
  }

  ## data component: exact Poisson count, events without replacement
  isSM <- identical(data, "SM")
  pb <- .passing_5d(pools, if (isSM) "toy" else data, nu_s, ratio_e, eta_edge)
  ib <- pick(pb, stats::rpois(1L, N_data * exp(nu_n) * pb$ratio), data)
  Xb <- pb$X[ib, , drop = FALSE]

  ## additive signal component
  Xs <- Xb[0, , drop = FALSE]
  if (!is.null(signal)) {
    ps <- .passing_5d(pools, signal, nu_s, ratio_e, eta_edge)
    ns <- if (is.null(N_sig)) stats::rpois(1L, NS * exp(nu_n) * ps$ratio) else as.integer(N_sig)
    if (ns > 0) Xs <- ps$X[pick(ps, ns, signal), , drop = FALSE]
  }

  D <- rbind(Xb, Xs)
  z <- c(rep(0L, nrow(Xb)), rep(1L, nrow(Xs)))
  o <- sample.int(nrow(D))
  D <- D[o, , drop = FALSE]; z <- z[o]; rownames(D) <- NULL

  ## reference: central, from the toy pool, disjoint from this toy's SM
  ## source events. Pre-selection sample Rpre = exactly round(NR_ratio * N0)
  ## events passing the cuts at nu = 0 plus the near-threshold events of
  ## the trimmed pool that fail at nu = 0, in the pool's own proportion, so
  ## that the pushforward T_nu Rpre also contains the events migrating INTO
  ## the acceptance. R = the passing-at-0 subset (what the fits use at nu = 0).
  ## Folds are stratified by pass0 so every fold has the same mix.
  rp    <- if (is.null(pools$ref)) "toy" else "ref"   # separate reference pool if present
  p0    <- .passing_5d(pools, rp, 0, ratio_e, eta_edge)
  n_tr  <- pools[[rp]]$n_trim
  is0   <- logical(n_tr); is0[p0$idx] <- TRUE
  used  <- logical(n_tr); if (!iid && isSM && rp == "toy") used[ib] <- TRUE
  a_pass <- which(is0 & !used); a_fail <- which(!is0 & !used)
  NR <- round(NR_ratio * N0)
  nF <- round(NR * (n_tr - length(p0$idx)) / length(p0$idx))
  if (!iid && (NR > length(a_pass) || nF > length(a_fail)))
    stop("draw_toy_5d: toy pool too small for NR_ratio = ", NR_ratio)
  ## stratified (pass / fail at nu = 0, proportional allocation) draw
  iR <- c(a_pass[sample.int(length(a_pass), NR, replace = iid)],
          a_fail[sample.int(length(a_fail), nF, replace = iid)])
  iR <- iR[sample.int(length(iR))]
  Rpre  <- pools[[rp]]$ev[iR, , drop = FALSE]; rownames(Rpre) <- NULL
  pass0 <- is0[iR]
  foldRpre <- integer(length(iR))
  foldRpre[pass0]  <- .make_folds5(sum(pass0), K)
  foldRpre[!pass0] <- .make_folds5(sum(!pass0), K)
  R <- Rpre[pass0, , drop = FALSE]; rownames(R) <- NULL

  ## auxiliary measurement
  nu_A <- nu_star
  fin  <- is.finite(sigma_A)
  if (any(fin)) nu_A[fin] <- stats::rnorm(sum(fin), nu_star[fin], sigma_A[fin])
  names(nu_A) <- names(nu_star) <- c("nu_n", "nu_s")

  list(x = features_5d(D, space), xR = features_5d(R, space),
       xn = features_5d(D, "raw5"), xnR = features_5d(R, "raw5"),
       wR = N0 / nrow(R), N = nrow(D), NR = nrow(R),
       fold = .make_folds5(nrow(D), K), foldR = foldRpre[pass0],
       Rpre = Rpre, pass0 = pass0, foldRpre = foldRpre,
       z = z, n_sig = sum(z), mll = D$mll, mllR = R$mll,
       nu_A = nu_A, nu_star = nu_star, sigma_A = sigma_A,
       N0 = N0, NR_ratio = NR_ratio, K = K, space = space,
       data = data, signal = signal, NS = NS)
}

## -------------------------------------------------------------
## 1.7 Delta-network training samples (NPLM Sect. 2.3, Eqs. 15-17)
## -------------------------------------------------------------
## One long table for the parametrized classifier
##   c(x; nu) = 1 / (1 + r(x; nu)),  r = exp(nu d1(x) + nu^2 d2(x) / 2),
##   L = sum_nu [ sum_{S0(nu)} w c^2 + sum_{S1(nu)} w (1 - c)^2 ],
## with columns: features (space; default raw5 = NPLM's inputs), nu,
## target (0 = S0(nu), 1 = S1), w, id (source-event row in the delta
## pool; the trainer splits train/validation by id). Preprocessing (NPLM/LMNP: eta
## standardized, pT divided by its mean) is left to the trainer.
## S0(nu_i): source events transformed at nu_i, then cut; S1: central,
## cut, repeated once per nu_i (NPLM's single S1 weighted by the number
## of nu points is the same loss). Weights normalize each sample to its
## expected count: sum_{S1} w = N0, sum_{S0(nu)} w = N(R_nu), so the
## acceptance change under the scale shift is carried by the deltas, as
## in NPLM. nu_n is handled analytically and does not enter.
## Normalization uses common random numbers: w0 = N0 / #S0-source events
## passing at nu = 0, so sum_{S0(nu)} w = N0 * pass(nu) / pass(0) on the
## SAME source events. With disjoint halves this removes the half-to-half
## count fluctuation (~1/sqrt(n)) from N(R_nu)/N(R0), which otherwise
## swamps the O(1e-3) acceptance effect at the small nu points.
##   paired = FALSE  S0 and S1 from disjoint halves of the delta pool
##                   (NPLM); TRUE uses all events for both (twice the
##                   size, correlated samples)
make_delta_samples_5d <- function(pools, nu_pts = CFG5$nu_delta, N0 = CFG5$N0,
                                  space = "raw5", paired = FALSE, seed = 1L,
                                  nu_check = c(0, -0.02, -0.015, -0.01, -0.006, -0.003,
                                               0.003, 0.006, 0.01, 0.015, 0.02),
                                  ratio_e = CFG5$ratio_e, eta_edge = CFG5$eta_edge) {
  stopifnot(inherits(pools, "pools_5d"), all(nu_pts != 0))
  set.seed(seed)
  P <- pools$delta; cuts <- pools$cuts; n <- P$n_trim
  if (paired) { s0 <- s1 <- seq_len(n) } else {
    s0 <- sort(sample.int(n, floor(n / 2))); s1 <- seq_len(n)[-s0]
  }
  X1 <- P$ev[s1, , drop = FALSE]; k1 <- pass_cuts_5d(X1, cuts)
  F1 <- features_5d(X1[k1, , drop = FALSE], space); id1 <- s1[k1]
  w1 <- N0 / sum(k1)
  w0 <- N0 / sum(pass_cuts_5d(P$ev[s0, , drop = FALSE], cuts))
  shifted <- function(nu) {
    X0 <- scale_nuisance_5d(P$ev[s0, , drop = FALSE], nu, ratio_e, eta_edge, pools$trim_log)
    k0 <- pass_cuts_5d(X0, cuts)
    list(F = features_5d(X0[k0, , drop = FALSE], space), id = s0[k0])
  }

  blocks <- vector("list", length(nu_pts)); NRnu <- numeric(length(nu_pts))
  for (i in seq_along(nu_pts)) {
    S0 <- shifted(nu_pts[i])
    NRnu[i] <- w0 * length(S0$id)
    blocks[[i]] <- rbind(
      data.frame(S0$F, nu = nu_pts[i], target = 0L, w = w0, id = S0$id),
      data.frame(F1,   nu = nu_pts[i], target = 1L, w = w1, id = id1))
  }
  out <- do.call(rbind, blocks)

  ## paired reconstruction-check sample: the S0 source events at each nu in
  ## nu_check (including 0), weights w0; log(n_nu / n_0) in any bin is then
  ## measured on the same events (NPLM Fig. 14, with common random numbers)
  chk <- NULL
  if (length(nu_check)) {
    chk <- do.call(rbind, lapply(nu_check, function(nu) {
      S <- shifted(nu); data.frame(S$F, nu = nu, w = w0, id = S$id) }))
  }
  attr(out, "N_R_nu") <- stats::setNames(NRnu, format(nu_pts))
  attr(out, "N_R0")   <- w1 * sum(k1)
  attr(out, "space")  <- space
  attr(out, "sizes")  <- c(S0_source = length(s0), S1_source = length(s1), S1_pass = sum(k1))
  attr(out, "check")  <- chk
  out
}

## CSV (gzip) for the Python delta-network trainer, plus <stem>_check.csv.gz
## (the paired reconstruction-check sample, if present) and <stem>_info.txt
## recording sizes and the acceptance effect log N(R_nu)/N(R0).
export_delta_samples_5d <- function(dsm, file) {
  file <- path.expand(file)
  dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
  wcsv <- function(df, f) { con <- gzfile(f, "w"); on.exit(close(con)); utils::write.csv(df, con, row.names = FALSE) }
  base <- dsm; attr(base, "check") <- NULL
  wcsv(base, file)
  side <- sub("\\.csv(\\.gz)?$", "", file)
  if (!is.null(attr(dsm, "check"))) wcsv(attr(dsm, "check"), paste0(side, "_check.csv.gz"))
  writeLines(c(sprintf("space: %s", attr(dsm, "space")),
               sprintf("sizes: %s", paste(names(attr(dsm, "sizes")), attr(dsm, "sizes"), sep = "=", collapse = ", ")),
               sprintf("N_R0: %.2f", attr(dsm, "N_R0")),
               sprintf("log N(R_nu)/N(R0) at nu = %s: %s",
                       paste(names(attr(dsm, "N_R_nu")), collapse = ", "),
                       paste(sprintf("%.5f", log(attr(dsm, "N_R_nu") / attr(dsm, "N_R0"))), collapse = ", "))),
             paste0(side, "_info.txt"))
  invisible(file)
}

## -------------------------------------------------------------
## 1.7b Import of the trained delta networks (train_delta_5d.py JSON)
## -------------------------------------------------------------
## The networks are evaluated in R by a plain forward pass (ReLU hidden
## layers, linear output) after the trainer's preprocessing
## z = (x - shift) / scale. load_delta_5d() verifies the R forward pass
## against the trainer's own predictions on its stored check set.
load_delta_5d <- function(file, tol = 1e-8) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("load_delta_5d: install.packages('jsonlite')")
  J  <- jsonlite::fromJSON(path.expand(file), simplifyVector = FALSE)
  rows2mat <- function(L) do.call(rbind, lapply(L, function(r) as.numeric(unlist(r))))
  net <- function(n) list(W = lapply(n$W, rows2mat), b = lapply(n$b, function(v) as.numeric(unlist(v))))
  dm <- list(features = unlist(J$features),
             shift = as.numeric(unlist(J$preproc$shift)), scale = as.numeric(unlist(J$preproc$scale)),
             d1 = net(J$nets$delta1), d2 = net(J$nets$delta2), meta = J$meta, file = file)
  class(dm) <- "delta_5d"
  Xc <- rows2mat(J$check$x); colnames(Xc) <- dm$features
  D  <- delta_hat_5d(dm, Xc)
  err <- max(abs(D[, "d1"] - unlist(J$check$delta1)), abs(D[, "d2"] - unlist(J$check$delta2)))
  if (!(err < tol)) stop("load_delta_5d: R forward pass differs from trainer by ", signif(err, 3))
  dm$check_err <- err
  dm
}

.mlp_forward <- function(net, Z) {
  h <- Z; L <- length(net$W)
  for (l in seq_len(L - 1L)) h <- pmax(sweep(h %*% net$W[[l]], 2L, net$b[[l]], "+"), 0)
  drop(h %*% net$W[[L]] + net$b[[L]])
}

## guard: functions taking the frozen delta networks fail early and clearly
## if handed something else (e.g. a delta-training table)
.check_dm <- function(dm, where = "") {
  if (!inherits(dm, "delta_5d"))
    stop(where, ": `dm` must be the delta networks from load_delta_5d(), not an object of class ",
         paste(class(dm), collapse = "/"), call. = FALSE)
  invisible(TRUE)
}

## d1, d2 at raw-feature rows xn (matrix with columns dm$features, e.g. toy$xn)
delta_hat_5d <- function(dm, xn) {
  .check_dm(dm, "delta_hat_5d")
  Z <- sweep(sweep(as.matrix(xn[, dm$features, drop = FALSE]), 2L, dm$shift, "-"), 2L, dm$scale, "/")
  cbind(d1 = .mlp_forward(dm$d1, Z), d2 = .mlp_forward(dm$d2, Z))
}

## log r(x; nu_s) = nu_s d1 + nu_s^2 d2 / 2 (shape + acceptance; nu_n is added separately)
log_r_5d <- function(dm, xn, nu_s, D = NULL) {
  if (is.null(D)) D <- delta_hat_5d(dm, xn)
  nu_s * D[, "d1"] + 0.5 * nu_s^2 * D[, "d2"]
}

print.delta_5d <- function(x, ...) {
  m <- x$meta
  cat(sprintf("delta_5d from %s\n  sizes %s; selected epoch %s of %s (%s); best val loss %.6g\n  nu points %s; R-vs-trainer max diff %.1e\n",
              basename(x$file), paste(unlist(m$sizes), collapse = "-"), m$selected_epoch, m$epochs, m$select,
              m$best_val_loss, paste(unlist(m$nu_points), collapse = ", "), x$check_err))
  invisible(x)
}

## -------------------------------------------------------------
## 1.8 Data checks
## -------------------------------------------------------------
## Layout and conventions of a loaded sample.
check_sample_5d <- function(X, cuts = CFG5$cuts, name = attr(X, "name")) {
  pc <- pass_cuts_5d(X, cuts)
  m5 <- sqrt(2 * X$pt1 * X$pt2 * (cosh(X$eta1 - X$eta2) - cos(X$delta_phi)))
  rel <- abs(m5 / X$mll - 1)
  cat(sprintf("%s: n = %d (dropped %s), pass cuts %d (acc %.4f)\n",
              if (is.null(name)) "sample" else name, nrow(X),
              format(attr(X, "n_dropped")), sum(pc), mean(pc)))
  cat(sprintf("  pt1 >= pt2: %s;  delta_phi range [%.3f, %.3f]\n",
              all(X$pt1 >= X$pt2), min(X$delta_phi), max(X$delta_phi)))
  cat(sprintf("  |eta| max: %.3f / %.3f\n", max(abs(X$eta1)), max(abs(X$eta2))))
  cat(sprintf("  mll vs massless formula: median |rel diff| %.2e, 99.9%% %.2e\n",
              stats::median(rel), stats::quantile(rel, .999)))
  q <- function(v) round(stats::quantile(v[pc], c(0, .001, .5, .999, 1)), 3)
  print(rbind(pt1 = q(X$pt1), pt2 = q(X$pt2), eta1 = q(X$eta1), eta2 = q(X$eta2),
              delta_phi = q(X$delta_phi), mll = q(X$mll)))
  invisible(list(acc = mean(pc), mll_rel = stats::quantile(rel, c(.5, .999))))
}

## NPLM Fig. 13 analogue: log N_b(nu_s)/N_b(0) in bins of `var` after
## cuts, over a grid of nu_s, with linear and quadratic fits on
## |nu_s| <= fit_max. Large quadratic residuals flag the regime where
## the second-order Taylor expansion of the deltas breaks down. Bins
## with fewer than min_count events at nu = 0 are reported as NA.
check_nuisance_bins_5d <- function(pools, pool = "delta", var = "pt1",
                                   breaks = c(20, 40, 60, 80, 100, 150, 200, 300, 500, 1000),
                                   nu_grid = seq(-0.02, 0.02, by = 0.0025), fit_max = 0.01,
                                   min_count = 50L,
                                   ratio_e = CFG5$ratio_e, eta_edge = CFG5$eta_edge) {
  P <- if (pool %in% c("delta", "toy")) pools[[pool]] else pools$sig[[pool]]
  cnt <- function(nu) {
    X <- scale_nuisance_5d(P$ev, nu, ratio_e, eta_edge, pools$trim_log)
    X <- X[pass_cuts_5d(X, pools$cuts), , drop = FALSE]
    tabulate(findInterval(X[[var]], breaks, left.open = TRUE), nbins = length(breaks) - 1L)
  }
  C0 <- cnt(0)
  L  <- sapply(nu_grid, function(nu) log(cnt(nu) / C0))
  L  <- matrix(L, nrow = length(C0))
  L[C0 < min_count, ] <- NA
  rownames(L) <- sprintf("(%g,%g]", utils::head(breaks, -1), breaks[-1])
  colnames(L) <- format(nu_grid)
  fit_res <- t(apply(L, 1L, function(y) {
    use <- abs(nu_grid) <= fit_max & is.finite(y)
    if (sum(use) < 3L) return(c(n0 = NA, slope_lin = NA, maxres_lin = NA, maxres_quad = NA))
    r1 <- stats::lm(y ~ 0 + nu, data.frame(y = y[use], nu = nu_grid[use]))
    r2 <- stats::lm(y ~ 0 + nu + I(nu^2), data.frame(y = y[use], nu = nu_grid[use]))
    c(n0 = NA, slope_lin = unname(stats::coef(r1)[1]),
      maxres_lin = max(abs(stats::resid(r1))), maxres_quad = max(abs(stats::resid(r2))))
  }))
  fit_res[, "n0"] <- C0
  list(logratio = L, fit = fit_res)
}

## =============================================================
## Part 2: per-fold fits
##   2.1 null fit (Theory_Methods.tex, Sect. "Extension: simulator-
##       accessed null", eq. ts-nullfit)
## =============================================================
##   nu^(-k) = argmax_nu  sum_{D_-k} h(X_i; nu)
##                       - w_-k sum_{R_-k} e^{h(X'_e; nu)} + l_A(nu),
##   h(x; nu) = nu_n + nu_s d1(x) + nu_s^2 d2(x) / 2   (frozen delta-hat),
##   w_-k = (K - 1) N0 / (K M_-k),
##   l_A(nu) = -sum_j (nu_j - nu_A,j)^2 / (2 sigma_j^2).
## The data enter only through N_-k, sum d1, sum d2. nu_n is profiled
## exactly (a concave 1-D equation); nu_s is maximized by safeguarded
## Newton on the profile likelihood, with analytic first and second
## derivatives (implicit-function correction for the profiled nu_n).
## With h quadratic in nu_s the profile need not be globally concave,
## so each fit reports: convergence, negative curvature at the optimum,
## and whether nu_s sits on the bound of the validated delta range.
## sigma = Inf: no constraint on that component; sigma = 0: fixed at nu_A.

## delta-hat for a toy's data and reference events (computed once per toy)
add_delta_5d <- function(toy, dm) {
  toy$dD <- delta_hat_5d(dm, toy$xn)
  toy$dR <- delta_hat_5d(dm, toy$xnR)
  toy
}

## Reference sums at nu_s: S0 = w sum E, S1 = w sum E g, S2 = w sum E (g^2 + d2),
## E = exp(nu_s d1 + nu_s^2 d2 / 2), g = dE/dnu_s / E = d1 + nu_s d2
.ref_sums_5d <- function(nu_s, dR, w) {
  g <- dR[, 1] + nu_s * dR[, 2]
  E <- exp(nu_s * dR[, 1] + 0.5 * nu_s^2 * dR[, 2])
  c(S0 = w * sum(E), S1 = w * sum(E * g), S2 = w * sum(E * (g * g + dR[, 2])))
}

## profiled nu_n for given reference sum S0: root of
##   N - e^v S0 - p_n (v - a_n) = 0   (strictly decreasing in v)
.profile_nun_5d <- function(N, S0, a_n, p_n, tol = 1e-12, maxit = 60L) {
  if (is.infinite(p_n)) return(a_n)
  if (p_n == 0) return(log(N / S0))
  v <- log(max(N, 0.5) / S0)
  for (it in seq_len(maxit)) {
    F  <- N - exp(v) * S0 - p_n * (v - a_n)
    dF <- -exp(v) * S0 - p_n
    st <- F / dF
    v  <- v - st
    if (abs(st) < tol) break
  }
  v
}

## Profile log-likelihood in nu_s with value, gradient and curvature.
.null_profile_5d <- function(nu_s, N, sd1, sd2, dR, w, a, p) {
  S  <- .ref_sums_5d(nu_s, dR, w)
  v  <- .profile_nun_5d(N, S[["S0"]], a[1], p[1])
  ev <- exp(v)
  pr_n <- if (is.finite(p[1])) 0.5 * p[1] * (v - a[1])^2 else 0
  ll <- N * v + nu_s * sd1 + 0.5 * nu_s^2 * sd2 - ev * S[["S0"]] - pr_n -
    0.5 * p[2] * (nu_s - a[2])^2
  gr <- sd1 + nu_s * sd2 - ev * S[["S1"]] - p[2] * (nu_s - a[2])
  Hss <- sd2 - ev * S[["S2"]] - p[2]
  Hns <- -ev * S[["S1"]]
  Hnn <- -ev * S[["S0"]] - if (is.finite(p[1])) p[1] else Inf
  cv <- Hss - if (is.finite(Hnn)) Hns^2 / Hnn else 0
  list(ll = ll, grad = gr, curv = cv, nu = c(nu_n = unname(v), nu_s = unname(nu_s)),
       hess = matrix(c(Hnn, Hns, Hns, Hss), 2, 2, dimnames = list(c("nu_n", "nu_s"), c("nu_n", "nu_s"))))
}

## Core fit on arrays.
##   N       training data count;  sd = c(sum d1, sum d2) over training data
##   dR      delta-hat (M x 2) on training reference events; w their weight
##   nu_A, sigma_A  auxiliary measurement (length 2, order nu_n, nu_s)
##   bounds  admissible nu_s range (validated range of delta-hat)
fit_null_5d <- function(N, sd, dR, w, nu_A, sigma_A,
                        bounds = c(-0.015, 0.015), nu_s0 = NULL,
                        tol = 1e-10, maxit = 50L, max_step = 2.5e-3) {
  sigma_A <- unname(as.numeric(sigma_A)); sd <- unname(as.numeric(sd)); dR <- unname(dR)
  p <- ifelse(is.infinite(sigma_A), 0, ifelse(sigma_A == 0, Inf, 1 / sigma_A^2))
  a <- unname(as.numeric(nu_A))
  prof <- function(s) .null_profile_5d(s, N, sd[1], sd[2], dR, w, a, p)
  if (is.infinite(p[2])) {                       # nu_s fixed by sigma_s = 0
    P <- prof(a[2])
    return(list(nu = P$nu, loglik = P$ll, grad = 0, curv = NA_real_, hess = P$hess,
                converged = TRUE, concave = NA, at_bound = FALSE, iter = 0L, method = "fixed"))
  }
  clamp <- function(s) min(max(s, bounds[1]), bounds[2])
  s <- clamp(if (is.null(nu_s0)) { if (is.finite(a[2])) a[2] else 0 } else nu_s0)
  P <- prof(s); conv <- FALSE; it <- 0L
  while (it < maxit) {
    it <- it + 1L
    step <- if (P$curv < 0) -P$grad / P$curv else sign(P$grad) * max_step
    step <- sign(step) * min(abs(step), max_step)
    s_new <- clamp(s + step)
    P_new <- prof(s_new); k <- 0L
    while (P_new$ll < P$ll - 1e-12 && k < 30L) {  # backtracking: never decrease ll
      step <- step / 2; s_new <- clamp(s + step); P_new <- prof(s_new); k <- k + 1L
    }
    moved <- abs(s_new - s)
    s <- s_new; P <- P_new
    at_b <- (s <= bounds[1] && P$grad < 0) || (s >= bounds[2] && P$grad > 0)
    if (moved < tol || abs(P$grad) * max(abs(step), 1e-12) < tol || at_b) { conv <- TRUE; break }
  }
  method <- "newton"
  if (!conv || !(P$curv < 0) && !at_b) {           # fallback: bracketed 1-D search
    o <- stats::optimize(function(s) prof(s)$ll, bounds, maximum = TRUE, tol = 1e-12)
    if (o$objective > P$ll) { s <- o$maximum; P <- prof(s) }
    method <- "newton+brent"; conv <- TRUE
  }
  at_b <- isTRUE(abs(s - bounds[1]) < 1e-12 || abs(s - bounds[2]) < 1e-12)
  list(nu = P$nu, loglik = P$ll, grad = P$grad, curv = P$curv, hess = P$hess,
       converged = conv, concave = P$curv < 0, at_bound = at_b, iter = it, method = method)
}

## Per-fold null fits for one toy (fold structure as in Theory_Methods.tex,
## eq. ts-weights): training data D_-k, training reference R_-k with weight
## w_-k = (K - 1) N0 / (K M_-k). Returns one row per fold plus the full-data
## fit (fold 0: all of D, all of R, w = N0 / M), which is a diagnostic only.
null_fits_5d <- function(toy, dm = NULL, bounds = c(-0.015, 0.015), ...) {
  if (is.null(toy$dD)) {
    if (is.null(dm)) stop("null_fits_5d: supply dm or call add_delta_5d() first")
    toy <- add_delta_5d(toy, dm)
  }
  K <- toy$K; N0 <- toy$N0
  one <- function(iD, iR, w) {
    f <- fit_null_5d(N = length(iD), sd = colSums(toy$dD[iD, , drop = FALSE]),
                     dR = toy$dR[iR, , drop = FALSE], w = w,
                     nu_A = toy$nu_A, sigma_A = toy$sigma_A, bounds = bounds, ...)
    c(nu_n = f$nu[["nu_n"]], nu_s = f$nu[["nu_s"]], loglik = f$loglik, grad = f$grad,
      curv = f$curv, converged = f$converged, concave = f$concave, at_bound = f$at_bound,
      iter = f$iter, fallback = f$method == "newton+brent", N_tr = length(iD), M_tr = length(iR), w = w)
  }
  rows <- lapply(seq_len(K), function(k) {
    iD <- which(toy$fold != k); iR <- which(toy$foldR != k)
    one(iD, iR, (K - 1) * N0 / (K * length(iR)))
  })
  full <- one(seq_len(toy$N), seq_len(toy$NR), N0 / toy$NR)
  out <- as.data.frame(do.call(rbind, c(list(full), rows)))
  out <- cbind(fold = 0:K, out)
  for (cc in c("converged", "concave", "at_bound", "fallback")) out[[cc]] <- as.logical(out[[cc]])
  attr(out, "toy") <- toy[c("nu_star", "nu_A", "sigma_A", "N", "NR", "K")]
  out
}

## Tangent directions u(x; nu) = d h / d nu = (1, d1 + nu_s d2) at rows with delta-hat D
tangent_5d <- function(D, nu_s) cbind(u_n = 1, u_s = D[, "d1"] + nu_s * D[, "d2"])

## Toy-level check of the null fit: bias and spread of the full-data and
## per-fold nu-hat at a true nu_star, and the frequency of flags. With
## sigma_A = c(Inf, Inf) nu-hat is the unconstrained MLE, which isolates
## delta-hat error (bias in nu_s-hat) from the auxiliary pull.
check_null_fit_5d <- function(pools, dm, n_toys = 100L, nu_star = c(0, 0),
                              sigma_A = CFG5$sigma_A, N0 = CFG5$N0, K = CFG5$K, seed = 1L, ...) {
  set.seed(seed)
  res <- lapply(seq_len(n_toys), function(b) {
    t  <- draw_toy_5d(pools, N0 = N0, nu_star = nu_star, sigma_A = sigma_A, K = K)
    nf <- null_fits_5d(t, dm, ...)
    cbind(toy = b, nf)
  })
  R  <- do.call(rbind, res)
  fu <- R[R$fold == 0, ]; fo <- R[R$fold > 0, ]
  pull <- function(x, x0) c(mean = mean(x - x0), sd = stats::sd(x), se_mean = stats::sd(x) / sqrt(length(x)))
  ## expected sd of the full-data nu_s-hat from the observed information
  info_sd <- with(fu, sqrt(-1 / curv))
  cat(sprintf("null fit check: %d toys, nu* = (%g, %g), sigma_A = (%g, %g), N0 = %g, K = %d\n",
              n_toys, nu_star[1], nu_star[2], sigma_A[1], sigma_A[2], N0, K))
  cat(sprintf("  full data  nu_s-hat - nu*_s: mean %+.2e (se %.1e), sd %.2e; mean sqrt(-1/curv) %.2e\n",
              pull(fu$nu_s, nu_star[2])[1], pull(fu$nu_s, nu_star[2])[3], pull(fu$nu_s, nu_star[2])[2],
              mean(info_sd)))
  cat(sprintf("  full data  nu_n-hat - nu*_n: mean %+.2e (se %.1e), sd %.2e\n",
              pull(fu$nu_n, nu_star[1])[1], pull(fu$nu_n, nu_star[1])[3], pull(fu$nu_n, nu_star[1])[2]))
  cat(sprintf("  per fold   nu_s-hat - nu*_s: mean %+.2e, sd %.2e\n",
              mean(fo$nu_s - nu_star[2]), stats::sd(fo$nu_s)))
  cat(sprintf("  flags (all fits): not converged %d, not concave %d, at bound %d, fallback used %d, of %d\n",
              sum(!R$converged), sum(!R$concave, na.rm = TRUE), sum(R$at_bound), sum(R$fallback), nrow(R)))
  invisible(R)
}

## -------------------------------------------------------------
## 2.1b Null fit by pushforward estimating equations (under test)
## -------------------------------------------------------------
## Same equations as the likelihood scores, with the model side computed
## by the exact nuisance map T_nu applied to the pre-selection reference
## instead of reweighting by e^{h-hat}:
##   count:  N_-k            = e^{nu_n} C(nu_s) + p_n (nu_n - nu_A,n)
##   scale:  sum_{D_-k} g    = e^{nu_n} G(nu_s) + p_s (nu_s - nu_A,s)
##   C(nu_s) = w sum_{Rpre_-k} 1[T_nu X' passes],
##   G(nu_s) = w sum_{Rpre_-k} g(T_nu X') 1[T_nu X' passes],
## with the fixed instrument g = delta-hat_1 and p = 1/sigma_A^2. Both
## sides have equal expectation at nu* for ANY fixed g, so nu-hat is
## consistent whatever delta-hat's error; delta-hat affects only the
## variance. nu_n is profiled exactly; nu_s is a 1-D root (G jumps by
## w g when a reference event crosses a cut, so Brent's method, not Newton).

.d1_5d <- function(dm, xn) {
  Z <- sweep(sweep(as.matrix(xn[, dm$features, drop = FALSE]), 2L, dm$shift, "-"), 2L, dm$scale, "/")
  .mlp_forward(dm$d1, Z)
}

## Matrix form of a pre-selection reference for fast repeated T_nu:
## raw columns, per-lepton scale exponents r_j (1 barrel, ratio_e endcap).
.push_prep_5d <- function(Rpre, ratio_e = CFG5$ratio_e, eta_edge = CFG5$eta_edge) {
  list(pt1 = Rpre$pt1, pt2 = Rpre$pt2, eta1 = Rpre$eta1, eta2 = Rpre$eta2,
       dphi = Rpre$delta_phi, mll = Rpre$mll,
       r1 = ifelse(abs(Rpre$eta1) >= eta_edge, ratio_e, 1),
       r2 = ifelse(abs(Rpre$eta2) >= eta_edge, ratio_e, 1), n = nrow(Rpre))
}

## T_nu applied to a prepared reference: canonical raw5 features of the
## events passing the cuts (same result as scale_nuisance_5d + pass_cuts_5d).
.push_apply_5d <- function(P, nu_s, cuts = CFG5$cuts) {
  f1 <- exp(nu_s * P$r1); f2 <- exp(nu_s * P$r2)
  a1 <- P$pt1 * f1; a2 <- P$pt2 * f2; m <- P$mll * sqrt(f1 * f2)
  hi <- pmax(a1, a2); lo <- pmin(a1, a2)
  k  <- lo > cuts$pt_min & hi < cuts$pt_max & m > cuts$mll_min &
        abs(P$eta1) < cuts$eta_max & abs(P$eta2) < cuts$eta_max
  sw <- (a2 > a1)[k]
  e1 <- P$eta1[k]; e2 <- P$eta2[k]; dp <- P$dphi[k]
  cbind(pt1 = hi[k], pt2 = lo[k], eta1 = ifelse(sw, e2, e1), eta2 = ifelse(sw, e1, e2),
        delta_phi = ifelse(sw, (2 * pi - dp) %% (2 * pi), dp))
}

## N, gD   training data count and sum of g = d1 over training data
## Rpre    pre-selection reference events (data frame), or a .push_prep_5d()
##         object; w their weight, normalized so that w * #(pass at nu = 0)
##         = expected training count at nu = 0
fit_null_push_5d <- function(N, gD, Rpre, w, nu_A, sigma_A, dm, bounds = c(-0.01, 0.01),
                             cuts = CFG5$cuts, ratio_e = CFG5$ratio_e, eta_edge = CFG5$eta_edge,
                             trim_log = CFG5$trim_log, tol = 1e-6, slope_h = 1e-3, slope = TRUE,
                             start = NULL, half_width = 1e-3) {
  if (max(abs(bounds)) * ratio_e > trim_log) stop("fit_null_push_5d: bounds exceed the pool trim window")
  sigma_A <- unname(as.numeric(sigma_A)); a <- unname(as.numeric(nu_A))
  p <- ifelse(is.infinite(sigma_A), 0, ifelse(sigma_A == 0, Inf, 1 / sigma_A^2))
  P <- if (is.data.frame(Rpre)) .push_prep_5d(Rpre, ratio_e, eta_edge) else Rpre
  n_eval <- 0L
  ev <- function(s) {
    n_eval <<- n_eval + 1L
    Xk <- .push_apply_5d(P, s, cuts)
    C <- w * nrow(Xk)
    G <- if (nrow(Xk)) w * sum(.d1_5d(dm, Xk)) else 0
    v <- .profile_nun_5d(N, C, a[1], p[1])
    Fs <- gD - exp(v) * G - (if (is.finite(p[2])) p[2] * (s - a[2]) else 0)
    c(F = Fs, nu_n = v, C = C, G = G)
  }
  if (is.infinite(p[2])) {
    r <- ev(a[2])
    return(list(nu = c(nu_n = r[["nu_n"]], nu_s = a[2]), F = 0, slope = NA_real_,
                at_bound = FALSE, iter = 0L, n_eval = n_eval, method = "fixed"))
  }
  ## bracket: around `start` (warm start, widened until F changes sign), else the bounds
  if (!is.null(start)) {
    hw <- half_width
    repeat {
      br <- c(max(start - hw, bounds[1]), min(start + hw, bounds[2]))
      lo <- ev(br[1]); hi <- ev(br[2])
      if (sign(lo[["F"]]) != sign(hi[["F"]]) || (br[1] == bounds[1] && br[2] == bounds[2])) break
      hw <- 4 * hw
    }
  } else {
    br <- bounds; lo <- ev(br[1]); hi <- ev(br[2])
  }
  it <- 0L
  if (sign(lo[["F"]]) == sign(hi[["F"]])) {           # no root inside: F is decreasing in nu_s
    s <- if (lo[["F"]] < 0) bounds[1] else bounds[2]; at_b <- TRUE
  } else {
    u <- stats::uniroot(function(s) ev(s)[["F"]], br, f.lower = lo[["F"]], f.upper = hi[["F"]],
                        tol = tol, maxiter = 200L)
    s <- u$root; it <- u$iter; at_b <- FALSE
  }
  r <- ev(s)
  sl <- if (slope) (ev(min(s + slope_h, bounds[2]))[["F"]] - ev(max(s - slope_h, bounds[1]))[["F"]]) /
          (min(s + slope_h, bounds[2]) - max(s - slope_h, bounds[1])) else NA_real_
  list(nu = c(nu_n = r[["nu_n"]], nu_s = s), F = r[["F"]], slope = sl,
       at_bound = at_b, iter = it, n_eval = n_eval, method = "push")
}

## Per-fold pushforward null fits for one toy (fold 0 = all data, diagnostic).
## Weight per pre-selection reference event: w_-k = (K-1) N0 / (K M0_-k), with
## M0_-k the number of training reference events passing at nu = 0.
null_fits_push_5d <- function(toy, dm, bounds = c(-0.01, 0.01), ...) {
  if (is.null(toy$Rpre)) stop("null_fits_push_5d: toy lacks Rpre (regenerate with the current draw_toy_5d)")
  if (is.null(toy$dD)) toy <- add_delta_5d(toy, dm)
  K <- toy$K; N0 <- toy$N0
  Pall <- .push_prep_5d(toy$Rpre)
  sub  <- function(P, i) { Q <- lapply(P[setdiff(names(P), "n")], `[`, i); Q$n <- length(i); Q }
  one <- function(iD, iR, w, start = NULL, slope = FALSE) {
    f <- fit_null_push_5d(N = length(iD), gD = sum(toy$dD[iD, "d1"]),
                          Rpre = if (length(iR) == Pall$n) Pall else sub(Pall, iR), w = w,
                          nu_A = toy$nu_A, sigma_A = toy$sigma_A, dm = dm, bounds = bounds,
                          start = start, slope = slope, ...)
    c(nu_n = f$nu[["nu_n"]], nu_s = f$nu[["nu_s"]], F = f$F, slope = f$slope,
      at_bound = f$at_bound, iter = f$iter, n_eval = f$n_eval)
  }
  ## full-data fit first (with the slope diagnostic), then warm-started folds
  full <- one(seq_len(toy$N), seq_len(Pall$n), N0 / sum(toy$pass0), slope = TRUE)
  rows <- lapply(seq_len(K), function(k) {
    iD <- which(toy$fold != k); iR <- which(toy$foldRpre != k)
    one(iD, iR, (K - 1) * N0 / (K * sum(toy$pass0[iR])), start = full[["nu_s"]])
  })
  out <- cbind(fold = 0:K, as.data.frame(do.call(rbind, c(list(full), rows))))
  out$at_bound <- as.logical(out$at_bound)
  out
}

## ---- accuracy tests (before adopting the pushforward fit) ----

## Population test with an INDEPENDENT reference: "data" = the whole toy
## pool transformed to nu*, "reference" = the (disjoint) ref pool at
## pre-selection (not the delta pool: delta-hat is fitted to those events,
## so evaluating the instrument there can be biased by overfitting). Both
## carry weights normalizing them to N0 at nu = 0.
## method = "push": pushforward equations; "ml": likelihood with h-hat
## (reference restricted to events passing at 0, as the ML fit uses).
## Remaining bias reflects only reference Monte Carlo noise (push) or
## additionally h-hat misspecification (ml).
pop_nu_indep_5d <- function(pools, dm, nu_s_star, method = c("push", "ml"), bounds = c(-0.01, 0.01)) {
  method <- match.arg(method)
  pD <- .passing_5d(pools, "toy", nu_s_star)
  wD <- CFG5$N0 / pools$toy$n_pass
  dD <- delta_hat_5d(dm, features_5d(pD$X[pD$idx, , drop = FALSE], "raw5"))
  if (is.null(pools$ref)) stop("pop_nu_indep_5d: needs a separate reference pool (make_pools_5d with frac_ref > 0)")
  Rp <- pools$ref$ev
  wR <- CFG5$N0 / pools$ref$n_pass
  if (method == "push") {
    f <- fit_null_push_5d(N = wD * nrow(dD), gD = wD * sum(dD[, "d1"]), Rpre = Rp, w = wR,
                          nu_A = c(0, 0), sigma_A = c(Inf, Inf), dm = dm, bounds = bounds)
  } else {
    R0 <- Rp[pass_cuts_5d(Rp, pools$cuts), , drop = FALSE]
    f <- fit_null_5d(N = wD * nrow(dD), sd = wD * colSums(dD), dR = delta_hat_5d(dm, features_5d(R0, "raw5")),
                     w = wR, nu_A = c(0, 0), sigma_A = c(Inf, Inf), bounds = bounds)
  }
  c(nu_s_star = nu_s_star, nu_s_hat = f$nu[["nu_s"]], ratio = if (nu_s_star != 0) f$nu[["nu_s"]] / nu_s_star else NA,
    bias = f$nu[["nu_s"]] - nu_s_star, nu_n_hat = f$nu[["nu_n"]])
}

## Toy test: ML and pushforward fits on the SAME toys. Reports bias and
## spread of nu_s-hat for both (full-data fit and per fold), their
## correlation, the efficiency ratio sd(push)/sd(ml), flags, and timing.
check_null_push_5d <- function(pools, dm, n_toys = 100L, nu_star = c(0, 0), sigma_A = CFG5$sigma_A,
                               N0 = CFG5$N0, K = CFG5$K, bounds = c(-0.01, 0.01), seed = 1L,
                               cores = 1L) {
  one_toy <- function(b) {
    set.seed((seed * 100003 + b) %% .Machine$integer.max)   # reproducible per toy, any core count
    t  <- add_delta_5d(draw_toy_5d(pools, N0 = N0, nu_star = nu_star, sigma_A = sigma_A, K = K), dm)
    t0 <- proc.time()[["elapsed"]]; ml <- null_fits_5d(t, bounds = bounds)
    t1 <- proc.time()[["elapsed"]]; pu <- null_fits_push_5d(t, dm, bounds = bounds)
    t2 <- proc.time()[["elapsed"]]
    data.frame(toy = b, fold = ml$fold, ml_s = ml$nu_s, ml_n = ml$nu_n,
               pu_s = pu$nu_s, pu_n = pu$nu_n, pu_bound = pu$at_bound, pu_slope = pu$slope,
               ml_conv = ml$converged & ml$concave & !ml$at_bound,
               t_ml = t1 - t0, t_pu = t2 - t1)
  }
  res <- if (cores > 1L) parallel::mclapply(seq_len(n_toys), one_toy, mc.cores = cores)
         else lapply(seq_len(n_toys), one_toy)
  bad <- vapply(res, inherits, logical(1), "try-error")
  if (any(bad)) stop("check_null_push_5d: ", sum(bad), " toys failed; first error: ", res[[which(bad)[1]]])
  R  <- do.call(rbind, res)
  t_ml <- sum(R$t_ml[R$fold == 0]); t_pu <- sum(R$t_pu[R$fold == 0])
  fu <- R[R$fold == 0, ]; fo <- R[R$fold > 0, ]
  s0 <- nu_star[2]; n0 <- nu_star[1]
  line <- function(lab, x, x0) cat(sprintf("  %-22s mean bias %+.2e (se %.1e), sd %.2e\n",
                                         lab, mean(x - x0), stats::sd(x) / sqrt(length(x)), stats::sd(x)))
  cat(sprintf("null fit ML vs pushforward: %d toys, nu* = (%g, %g), sigma_A = (%g, %g), N0 = %g, K = %d\n",
              n_toys, n0, s0, sigma_A[1], sigma_A[2], N0, K))
  line("ML   nu_s (full data)", fu$ml_s, s0); line("PUSH nu_s (full data)", fu$pu_s, s0)
  line("ML   nu_n (full data)", fu$ml_n, n0); line("PUSH nu_n (full data)", fu$pu_n, n0)
  line("ML   nu_s (per fold)", fo$ml_s, s0);  line("PUSH nu_s (per fold)", fo$pu_s, s0)
  cat(sprintf("  efficiency sd(push)/sd(ml), full data: %.3f;  cor(push, ml): %.3f\n",
              stats::sd(fu$pu_s) / stats::sd(fu$ml_s), stats::cor(fu$pu_s, fu$ml_s)))
  cat(sprintf("  flags: push at bound %d, ML not clean %d, of %d fits;  slope of F < 0 in %.1f%% of full-data fits\n",
              sum(R$pu_bound), sum(!R$ml_conv), nrow(R), 100 * mean(fu$pu_slope < 0)))
  cat(sprintf("  time per toy (per core): ML %.2fs, push %.2fs\n", t_ml / n_toys, t_pu / n_toys))
  invisible(R)
}
## -------------------------------------------------------------
## 2.2 Witness dictionary (fixed; built once from the pool)
## -------------------------------------------------------------
## The 4-D analogue of utils.R::make_dict. The dictionary is a FIXED set
## of functions, constructed once from SM events of the toy pool passing
## the cuts at nu = 0 (the base measure f0), so it never depends on a
## toy, a fold or a fit. Blocks, in Gram-Schmidt order under f0:
##   constant (glmnet intercept; not a column)
##   tangent  {d1-hat, d2-hat}, orthonormalized, unpenalized: their span
##            contains log n_nu - log n_0 to second order at every nu
##   smooth   tensor Legendre products, total degree 1..smooth_deg, in
##            v_j = 2 u_j - 1 with u_j the f0-CDF of the smooth coordinates.
##            space = "5d" (default; NPLM's five features, no reduction):
##            (pt1 + pt2, pt2/pt1, eta1 - eta2, eta1 + eta2, cos dphi), 55
##            atoms at degree 3. space = "4d" (boost-invariant ablation):
##            (pt1 + pt2, pt2 / pt1, cosh deta, cos dphi):
##            the same 4-D space as dict4, decorrelated (with u_pt1, u_pt2
##            directly, cor 0.99, the raw design had kappa ~ 2.5e6 and
##            atoms up to ~40 in empty corners; with sum/ratio kappa ~ 3e2);
##            orthogonal to 1 and tangent, unpenalized (34 atoms at degree 3)
##   bump     Gaussians exp(-|u_pt - c|^2 / (2 w^2)) in the (u_pt1, u_pt2)
##            plane, widths `bump_widths`, centres on a square grid with
##            spacing spacing_rel * w; bumps with expected count below
##            min_events (under N0) are dropped; projected against
##            1, tangent and smooth, unit variance, penalized
## Penalties: tangent free (nuisance directions); smooth penalized with
## factor pf_smooth relative to the bumps (its lambda is lambda_bump *
## pf_smooth); bumps factor 1. Default width 0.1: on the Z' benchmark it
## gives z_opt 6.60 vs 6.93 (w = 0.05, 338 atoms) and 6.17 (w = 0.2) with
## 96 atoms (a signal-informed choice; see Experiments.md).
## Off-sample evaluation reproduces the construction exactly: raw columns
## [1, d1, d2, Legendre products] times the stored R^-1, then the bumps
## minus their stored projection coefficients, divided by their sd.

.legendre5 <- function(x, deg) switch(as.character(deg), "0" = rep(1, length(x)), "1" = x,
                                      "2" = (3 * x^2 - 1) / 2, "3" = (5 * x^3 - 3 * x) / 2,
                                      "4" = (35 * x^4 - 30 * x^2 + 3) / 8,
                                      stop(".legendre5: degree > 4"))

## dict4 coordinates from raw5 rows (matrix with columns pt1, pt2, eta1, eta2, delta_phi)
.dict4_from_raw5 <- function(X5) cbind(pt1 = X5[, "pt1"], pt2 = X5[, "pt2"],
                                       ch_deta = cosh(X5[, "eta1"] - X5[, "eta2"]),
                                       c_dphi = cos(X5[, "delta_phi"]))
## smooth-block coordinates: a bijection of dict4 with decorrelated pT's
.smooth4_from_dict4 <- function(Z) cbind(pt_sum = Z[, 1] + Z[, 2], pt_ratio = Z[, 2] / Z[, 1],
                                         ch_deta = Z[, 3], c_dphi = Z[, 4])
## smooth-block coordinates of the witness in NPLM's 5-D feature space
## (default): decorrelated pT's, the eta's unfolded as (difference, sum),
## and cos(dphi) for the periodicity of dphi. No symmetry is imposed.
.smooth5_from_raw5 <- function(X5) cbind(pt_sum = X5[, "pt1"] + X5[, "pt2"],
                                         pt_ratio = X5[, "pt2"] / X5[, "pt1"],
                                         deta = X5[, "eta1"] - X5[, "eta2"],
                                         seta = X5[, "eta1"] + X5[, "eta2"],
                                         c_dphi = cos(X5[, "delta_phi"]))
## "5d_fold": the 4-D coordinates plus eta1 + eta2, i.e. 5-D with deta
## entering through cosh (witness even in deta at fixed eta1 + eta2)
.smooth5f_from_raw5 <- function(X5) cbind(.smooth4_from_dict4(.dict4_from_raw5(X5)),
                                          seta = X5[, "eta1"] + X5[, "eta2"])
.smooth_coords_5d <- function(X5, space) switch(space,
  "4d" = .smooth4_from_dict4(.dict4_from_raw5(X5)),
  "5d" = .smooth5_from_raw5(X5),
  "5d_fold" = .smooth5f_from_raw5(X5),
  stop("unknown space ", space))

## monotone piecewise-linear CDF from n_knots quantiles; values clamp to [0, 1]
.cdf_fun5 <- function(z, n_knots = 2001L) {
  p <- seq(0, 1, length.out = n_knots)
  q <- stats::quantile(z, p, names = FALSE, type = 7)
  keep <- !duplicated(q)
  stats::approxfun(q[keep], p[keep], rule = 2, ties = "ordered")
}

.dict5_raw <- function(dict, X5, D = NULL) {
  if (is.null(D)) D <- delta_hat_5d(dict$dm, X5)
  Zs <- .smooth_coords_5d(X5, dict$space %||% "4d")
  U <- .bump_U_5d(dict, X5)                                               # bump coordinates
  V <- 2 * sapply(seq_along(dict$cdf), function(j) dict$cdf[[j]](Zs[, j])) - 1
  V <- matrix(V, ncol = length(dict$cdf))
  P <- vapply(seq_len(nrow(dict$multi)), function(r) {
    m <- dict$multi[r, ]; out <- rep(1, nrow(V))
    for (j in which(m > 0)) out <- out * .legendre5(V[, j], m[j])
    out
  }, numeric(nrow(V)))
  list(raw = cbind(1, D[, "d1"], D[, "d2"], P), U = U)
}

## bump coordinates: f0-CDFs of pt1, pt2 and, for 3-D bumps, cosh(eta1 - eta2)
.bump_raw_coords_5d <- function(X5, coords) {
  vapply(coords, function(v) switch(v, pt1 = X5[, "pt1"], pt2 = X5[, "pt2"],
                                   ch_deta = cosh(X5[, "eta1"] - X5[, "eta2"]),
                                   stop("unknown bump coordinate ", v)), numeric(nrow(X5)))
}
.bump_U_5d <- function(dict, X5) {
  co <- dict$bump_coords %||% c("pt1", "pt2")
  R  <- matrix(.bump_raw_coords_5d(X5, co), ncol = length(co))
  vapply(seq_along(co), function(j) dict$cdf_b[[j]](R[, j]), numeric(nrow(X5)))
}
## Gaussian bumps, axis-aligned: exp(-sum_d (u_d - c_jd)^2 / (2 w_jd^2)); bump_w is a
## J x d matrix of per-dimension widths (a vector = isotropic, the 2-D layout)
.dict5_bumps_raw <- function(dict, U) {
  U <- matrix(U, nrow = NROW(U)); d <- ncol(U)
  W <- if (is.null(dim(dict$bump_w))) matrix(dict$bump_w, length(dict$bump_w), d) else dict$bump_w
  q <- 0
  for (j in seq_len(d)) q <- q + sweep(outer(U[, j], dict$bump_c[, j], "-"), 2L, W[, j], "/")^2
  exp(-q / 2)
}

## bump_coords = c("pt1", "pt2") (default, 2-D bumps) or c("pt1", "pt2", "ch_deta")
## (3-D bumps, also localized in cosh(eta1 - eta2), so that they can follow the
## mll curve mll ~ 2 pT cosh(deta/2)); bump_width3 is the width (and spacing) in
## the third coordinate.
make_dict_5d <- function(pools, dm, space = c("5d", "4d", "5d_fold"), n_base = 50000L, smooth_deg = 3L,
                         bump_widths = 0.1, spacing_rel = 1, pf_smooth = 1,
                         bump_coords = c("pt1", "pt2"), bump_width3 = 0.2,
                         min_events = 10, N0 = CFG5$N0, seed = 11L) {
  space <- match.arg(space); .check_dm(dm, "make_dict_5d")
  set.seed(seed)
  p0 <- .passing_5d(pools, "toy", 0)
  ib <- p0$idx[sample.int(length(p0$idx), min(n_base, length(p0$idx)))]
  X5 <- features_5d(p0$X[ib, , drop = FALSE], "raw5")
  Zs <- .smooth_coords_5d(X5, space); nc <- ncol(Zs)
  cdf   <- lapply(seq_len(nc), function(j) .cdf_fun5(Zs[, j]))
  Rb    <- matrix(.bump_raw_coords_5d(X5, bump_coords), ncol = length(bump_coords))
  cdf_b <- lapply(seq_along(bump_coords), function(j) .cdf_fun5(Rb[, j]))
  g <- as.matrix(expand.grid(rep(list(0:smooth_deg), nc)))
  multi <- g[rowSums(g) >= 1 & rowSums(g) <= smooth_deg, , drop = FALSE]
  multi <- multi[order(rowSums(multi)), , drop = FALSE]; dimnames(multi) <- NULL
  dict <- list(dm = dm, space = space, smooth_coords = colnames(Zs), cdf = cdf, cdf_b = cdf_b, multi = multi,
               bump_coords = bump_coords)

  ## [1 | tangent | smooth]: orthonormal under the base (equal weights 1/n)
  D  <- delta_hat_5d(dm, X5)
  rw <- .dict5_raw(dict, X5, D)
  n  <- nrow(X5)
  qrX <- qr(rw$raw / sqrt(n))
  if (qrX$rank < ncol(rw$raw)) stop("make_dict_5d: raw design rank ", qrX$rank, " < ", ncol(rw$raw))
  piv <- qrX$pivot
  if (!identical(piv, seq_len(ncol(rw$raw)))) stop("make_dict_5d: unexpected pivoting in QR")
  Rinv <- backsolve(qr.R(qrX), diag(ncol(rw$raw)))
  Q <- rw$raw %*% Rinv                                    # Q'Q / n = I
  if (mean(Q[, 1]) < 0) { Q[, 1] <- -Q[, 1]; Rinv[, 1] <- -Rinv[, 1] }

  ## bumps: grid per width, drop the empty ones
  cs <- do.call(rbind, lapply(bump_widths, function(w) {
    s <- spacing_rel * w; cc <- seq(s / 2, 1 - s / 2, by = s)
    cbind(as.matrix(expand.grid(c1 = cc, c2 = cc)), w = w)
  }))
  dict$bump_c <- cs[, 1:2, drop = FALSE]; dict$bump_w <- cs[, 3]
  if (length(bump_coords) == 3L) {               # replicate the plane grid across the third coordinate
    c3 <- seq(bump_width3 / 2, 1 - bump_width3 / 2, by = bump_width3)
    g  <- expand.grid(i = seq_len(nrow(cs)), c3 = c3)
    dict$bump_c <- cbind(cs[g$i, 1:2, drop = FALSE], c3 = g$c3)
    dict$bump_w <- cbind(w1 = cs[g$i, 3], w2 = cs[g$i, 3], w3 = bump_width3)
  }
  Braw <- .dict5_bumps_raw(dict, rw$U)
  keep <- N0 * colMeans(Braw) >= min_events
  dict$bump_c <- dict$bump_c[keep, , drop = FALSE]
  dict$bump_w <- if (is.null(dim(dict$bump_w))) dict$bump_w[keep] else dict$bump_w[keep, , drop = FALSE]
  Braw <- Braw[, keep, drop = FALSE]
  Cb   <- crossprod(Q, Braw) / n                          # projection on [1 | tangent | smooth]
  Bp   <- Braw - Q %*% Cb
  sdB  <- sqrt(colMeans(Bp^2))
  if (any(sdB < 1e-6)) stop("make_dict_5d: degenerate bump(s)")

  p_tan <- 2L; p_sm <- nrow(multi); J <- sum(keep)
  dict <- c(dict, list(
    Rinv = Rinv, Cb = Cb, sdB = sdB, n_base = n, N0 = N0,
    p_tan = p_tan, p_smooth = p_sm, J = J,
    idx_tan = seq_len(p_tan), idx_smooth = p_tan + seq_len(p_sm),
    idx_bump = p_tan + p_sm + seq_len(J),
    pen_factor = c(rep(0, p_tan), rep(pf_smooth, p_sm), rep(1, J)), pf_smooth = pf_smooth,
    bump_mass = colMeans(Braw), smooth_deg = smooth_deg, bump_widths = bump_widths,
    spacing_rel = spacing_rel, min_events = min_events, seed = seed,
    base_idx = ib))
  dict$idx_unpen <- dict$idx_tan                          # unpenalized columns
  dict$idx_ts    <- c(dict$idx_tan, dict$idx_smooth)       # orthonormal blocks
  class(dict) <- "dict_5d"
  dict
}

## Atoms at raw5 rows X5 (optionally with precomputed delta-hat D):
## [tangent (2) | smooth (p_smooth) | bumps (J)], constant excluded.
eval_atoms_5d <- function(dict, X5, D = NULL) {
  rw <- .dict5_raw(dict, X5, D)
  Q  <- rw$raw %*% dict$Rinv
  Bstd <- sweep(.dict5_bumps_raw(dict, rw$U) - Q %*% dict$Cb, 2L, dict$sdB, "/")
  A <- cbind(Q[, -1L, drop = FALSE], Bstd)
  colnames(A) <- c("T1", "T2", paste0("P", seq_len(dict$p_smooth)), paste0("B", seq_len(dict$J)))
  A
}

print.dict_5d <- function(x, ...) {
  cat(sprintf("dict_5d [%s witness]: %d tangent + %d smooth (degree <= %d in %s; penalty factor %g) + %d bumps (widths %s, spacing %.2f w); base n = %d\n",
              x$space %||% "4d", x$p_tan, x$p_smooth, x$smooth_deg,
              paste(x$smooth_coords %||% c("pt_sum", "pt_ratio", "ch_deta", "c_dphi"), collapse = ", "),
              x$pf_smooth, x$J, paste(x$bump_widths, collapse = "/"), x$spacing_rel, x$n_base))
  co <- x$bump_coords %||% c("pt1", "pt2")
  w1 <- if (is.null(dim(x$bump_w))) x$bump_w else x$bump_w[, 1]
  tw <- table(w1)
  cat(sprintf("  bumps in (%s)%s; per plane width: %s\n", paste0("u_", co, collapse = ", "),
              if (length(co) == 3L) sprintf(", width %g in u_%s", x$bump_w[1, 3], co[3]) else "",
              paste(names(tw), tw, sep = ": ", collapse = ", ")))
  invisible(x)
}

## Checks on an INDEPENDENT sample of pool events passing at nu = 0:
## orthonormality (Gram vs I for tangent/smooth; unit variance for all;
## zero means), conditioning per block, and the largest atom values.
check_dict_5d <- function(dict, pools, n_check = 50000L, seed = 12L) {
  set.seed(seed)
  p0 <- .passing_5d(pools, "toy", 0)
  cand <- setdiff(p0$idx, dict$base_idx)
  X5 <- features_5d(p0$X[cand[sample.int(length(cand), min(n_check, length(cand)))], , drop = FALSE], "raw5")
  A  <- eval_atoms_5d(dict, X5)
  G  <- crossprod(A) / nrow(A)
  iu <- dict$idx_ts; ib <- dict$idx_bump
  kap <- function(M) { e <- eigen(M, symmetric = TRUE, only.values = TRUE)$values; max(e) / min(e) }
  out <- c(n = nrow(A),
           max_abs_mean       = max(abs(colMeans(A))),
           unpen_gram_vs_I    = max(abs(G[iu, iu] - diag(length(iu)))),
           max_abs_var_minus1 = max(abs(diag(G) - 1)),
           unpen_vs_bump      = max(abs(G[iu, ib])),
           kappa_unpen        = kap(G[iu, iu]),
           min_eig_bump       = min(eigen(G[ib, ib], symmetric = TRUE, only.values = TRUE)$values),
           max_abs_smooth     = max(abs(A[, dict$idx_smooth])),
           max_abs_tangent    = max(abs(A[, dict$idx_tan])),
           max_abs_bump       = max(abs(A[, ib])))
  print(signif(out, 4))
  invisible(out)
}

## Signal coverage: for each atom, the expected standardized score of a
## signal of NS events, z_j = NS * E_S[A_j] / sqrt(N0) (atoms have mean 0
## and variance 1 under f0, so this is the single-atom Poisson z). Compare
## the best bump with the oracle mll window (NPLM-style cut and count).
signal_coverage_5d <- function(dict, pools, signal = "Zprime300", NS = 120, N0 = CFG5$N0,
                               n_sig = 20000L, window = c(278, 322), top = 10L, seed = 13L) {
  set.seed(seed)
  ps <- .passing_5d(pools, signal, 0)
  is <- ps$idx[sample.int(length(ps$idx), min(n_sig, length(ps$idx)))]
  XS <- features_5d(ps$X[is, , drop = FALSE], "raw5")
  A  <- eval_atoms_5d(dict, XS)
  z  <- NS * colMeans(A) / sqrt(N0)
  ib <- dict$idx_bump
  ord <- ib[order(-abs(z[ib]))][seq_len(min(top, length(ib)))]
  p0 <- .passing_5d(pools, "toy", 0)
  fS <- mean(ps$X$mll[ps$idx] > window[1] & ps$X$mll[ps$idx] < window[2])
  fB <- mean(p0$X$mll[p0$idx] > window[1] & p0$X$mll[p0$idx] < window[2])
  cat(sprintf("coverage of %s (NS = %g, N0 = %g):\n", signal, NS, N0))
  cat(sprintf("  oracle mll window [%g, %g]: s = %.1f, b = %.1f, s/sqrt(b) = %.2f\n",
              window[1], window[2], NS * fS, N0 * fB, NS * fS / sqrt(N0 * fB)))
  cat(sprintf("  best single bump z = %.2f; best smooth z = %.2f; tangent z = %s\n",
              max(abs(z[ib])), max(abs(z[dict$idx_smooth])),
              paste(sprintf("%.2f", z[dict$idx_tan]), collapse = ", ")))
  ## best linear combination within a block set: z_opt = sqrt(z' G^-1 z), G the f0-Gram
  ## (estimated on pool events); this is the single-direction z of the oracle witness
  ## restricted to that span, i.e. an upper bound on what the dictionary can deliver
  X0 <- features_5d(p0$X[p0$idx[sample.int(length(p0$idx), min(50000L, length(p0$idx)))], , drop = FALSE], "raw5")
  A0 <- eval_atoms_5d(dict, X0); G0 <- crossprod(A0) / nrow(A0)
  zopt <- function(idx) { e <- eigen(G0[idx, idx], symmetric = TRUE); k <- e$values > 1e-8 * max(e$values)
    v <- crossprod(e$vectors[, k, drop = FALSE], z[idx]); sqrt(sum(v^2 / e$values[k])) }
  iu <- dict$idx_ts
  cat(sprintf("  z_opt: tangent+smooth %.2f;  + top-%d bumps %.2f;  + all bumps %.2f\n",
              zopt(iu), top, zopt(c(iu, ord)), zopt(c(iu, ib))))
  jj <- ord - min(ib) + 1
  print(data.frame(atom = colnames(A)[ord], round(dict$bump_c[jj, , drop = FALSE], 3),
                   w = if (is.null(dim(dict$bump_w))) dict$bump_w[jj] else dict$bump_w[jj, 1],
                   z = round(z[ord], 2)), row.names = FALSE)
  invisible(z)
}

## -------------------------------------------------------------
## 2.3 Null representation per fold (W1)
## -------------------------------------------------------------
## The fitted null at nu-hat^(-k) as weighted reference events, the object
## the alternative fit and the test consume (the 5-D analogue of the 1-D
## quadrature grid with weights expo * wp). Weights follow
## Theory_Methods.tex eq. ts-weights: w_-k = (K-1) N0 / (K M0_-k) for the
## training reference, w_k = N0 / (K M0_k) for the test reference, with M0
## the number of reference events passing at nu = 0.
##   method "push": points = T_nu-hat applied to the pre-selection reference,
##            keeping those that pass; weight w e^{nu_n} (exact family)
##   method "ml":   points = reference passing at nu = 0; weight
##            w exp(h-hat(x; nu-hat)) (delta-hat reweighting; comparison)
## Test-base derivatives with respect to nu_s are stored as the atoms at
## nu_s +- eps (push: transformed points and their weights; ml: same points,
## reweighted), so that d/dnu_s of E_nu[phi] is a central difference for any
## linear witness phi = a + A beta. The derivative in nu_n is E_nu[phi]
## itself. For the null-fit Jacobian (the sandwich of the pushforward
## estimating equations) the training-base count C and instrument sum G
## are stored at nu_s and nu_s +- eps.
null_rep_5d <- function(toy, k, nu, dict, method = c("push", "ml"), eps = 1e-3,
                        parts = c("all", "tr")) {
  parts <- match.arg(parts)
  method <- match.arg(method)
  K <- toy$K; N0 <- toy$N0; dm <- dict$dm
  nu_n <- unname(nu[1]); nu_s <- unname(nu[2])
  iR_tr <- if (k == 0) seq_len(nrow(toy$Rpre)) else which(toy$foldRpre != k)
  iR_te <- if (k == 0) seq_len(nrow(toy$Rpre)) else which(toy$foldRpre == k)
  M0_tr <- sum(toy$pass0[iR_tr]); M0_te <- sum(toy$pass0[iR_te])
  w_tr <- (if (k == 0) 1 else (K - 1) / K) * N0 / M0_tr
  w_te <- (if (k == 0) 1 else 1 / K) * N0 / M0_te
  ## src: position (within iR) of the pre-selection event behind each base point
  base <- function(iR, w, s) {
    if (method == "push") {
      P  <- .push_prep_5d(toy$Rpre[iR, , drop = FALSE])
      f1 <- exp(s * P$r1); f2 <- exp(s * P$r2)
      cuts <- CFG5$cuts
      kk <- pmin(P$pt1 * f1, P$pt2 * f2) > cuts$pt_min & pmax(P$pt1 * f1, P$pt2 * f2) < cuts$pt_max &
            P$mll * sqrt(f1 * f2) > cuts$mll_min & abs(P$eta1) < cuts$eta_max & abs(P$eta2) < cuts$eta_max
      X5 <- .push_apply_5d(P, s)
      D  <- delta_hat_5d(dm, X5)
      list(X5 = X5, D = D, omega = rep(w * exp(nu_n), nrow(X5)), src = which(kk))
    } else {
      src <- which(toy$pass0[iR])
      R0 <- toy$Rpre[iR[src], , drop = FALSE]
      X5 <- features_5d(R0, "raw5"); D <- delta_hat_5d(dm, X5)
      list(X5 = X5, D = D, omega = w * exp(nu_n + s * D[, "d1"] + 0.5 * s^2 * D[, "d2"]), src = src)
    }
  }
  tr  <- base(iR_tr, w_tr, nu_s)
  tr$A <- eval_atoms_5d(dict, tr$X5, tr$D)
  if (parts == "tr")
    return(list(method = method, k = k, nu = c(nu_n = nu_n, nu_s = nu_s), w_tr = w_tr,
                tr = tr, expo_tr = sum(tr$omega)))
  te  <- base(iR_te, w_te, nu_s)
  tep <- base(iR_te, w_te, nu_s + eps); tem <- base(iR_te, w_te, nu_s - eps)
  te$A  <- eval_atoms_5d(dict, te$X5, te$D)
  tep$A <- eval_atoms_5d(dict, tep$X5, tep$D); tem$A <- eval_atoms_5d(dict, tem$X5, tem$D)
  ## null-fit Jacobian ingredients on the training base (instrument g = d1)
  CG <- function(b) c(C = sum(b$omega), G = sum(b$omega * b$D[, "d1"]))
  trp <- base(iR_tr, w_tr, nu_s + eps); trm <- base(iR_tr, w_tr, nu_s - eps)
  list(method = method, k = k, nu = c(nu_n = nu_n, nu_s = nu_s), eps = eps,
       w_tr = w_tr, w_te = w_te, M_te = length(iR_te), M0_te = M0_te, te_pass0 = toy$pass0[iR_te],
       tr = tr, te = te, te_p = tep[c("A", "omega")], te_m = tem[c("A", "omega")],
       CG = rbind(m = CG(trm), c = CG(tr), p = CG(trp)),
       expo_tr = sum(tr$omega), expo_te = sum(te$omega))
}

## E_nu-hat[phi] on the test base and its nu_s-derivative, for phi = a + A beta
.null_mean_5d <- function(rep, a, beta) {
  E <- function(b) sum(b$omega * (a + as.vector(b$A %*% beta)))
  c(mean = E(rep$te), d_nu_n = E(rep$te), d_nu_s = (E(rep$te_p) - E(rep$te_m)) / (2 * rep$eps))
}

## -------------------------------------------------------------
## 2.4 Alternative fit per fold (W3)
## -------------------------------------------------------------
## Penalized Poisson log-likelihood ratio of the training data against the
## training base (Theory_Methods.tex eq. ts-altfit):
##   max_beta  sum_{D_-k} f(X_i) - sum_b omega_b (e^{f(x_b)} - 1) - lambda sum_j pf_j |beta_j|,
## f = a + A beta. glmnet Poisson on pooled rows (Berman-Turner): base rows
## response 0, weight omega_b; data rows response 1/eps_bt, weight eps_bt
## (each contributes f - eps_bt e^f). glmnet normalizes the weights to sum
## to nobs, so its objective is -(1/W) sum w l + lambda_g sum pf |beta|
## with W = sum of weights; the KKT condition |score_j| <= lambda_g W pf_j
## with z_j = score_j / sqrt(expo_tr) gives lambda_g = lambda_z sqrt(expo_tr) / W.
## glmnet also rescales penalty.factor internally to sum to the number of
## columns p; the effective penalty is lambda_g (p / sum pf) sum pf_j |beta_j|,
## so lambda_g is multiplied by sum(pf) / p to undo it. Atom j then enters
## when |z_j| > lambda_z pf_j: bumps at lambda_z, smooth at lambda_z * pf_smooth.
## lambda_z is the bump penalty in z-units; the smooth block's is
## lambda_z * pf_smooth (dict$pen_factor). The witness is a + f (pois mode:
## log intensity ratio, no normalization).
fit_witness_5d <- function(rep, A_D, lambda_z, dict, eps_bt = 1e-5, n_path = 5L,
                           thresh = 1e-10, maxit = 1e5, polish = TRUE) {
  if (!requireNamespace("glmnet", quietly = TRUE)) stop("fit_witness_5d: package 'glmnet' is required")
  nb <- nrow(rep$tr$A); nD <- nrow(A_D)
  X  <- rbind(rep$tr$A, A_D)
  y  <- c(rep(0, nb), rep(1 / eps_bt, nD))
  wt <- c(rep$tr$omega, rep(eps_bt, nD))
  expo <- rep$expo_tr; W <- sum(wt)
  lam <- lambda_z * sqrt(expo) / W * sum(dict$pen_factor) / length(dict$pen_factor)
  lam_path <- lam * exp(seq(log(4), 0, length.out = n_path))
  fit <- glmnet::glmnet(X, y, family = "poisson", weights = wt, penalty.factor = dict$pen_factor,
                        lambda = lam_path, standardize = FALSE, intercept = TRUE,
                        thresh = thresh, maxit = maxit)
  beta <- as.vector(fit$beta[, n_path]); a <- unname(fit$a0[n_path])
  ## polish: exact KKT on glmnet's active set with fixed signs. glmnet's
  ## convergence test is relative to a null deviance dominated by the
  ## Berman-Turner data rows, which leaves |z_j| - lambda_z ~ 0.04 on the
  ## active bumps; a few Newton steps on the (smooth, concave) restricted
  ## objective  sum_D f - sum_b omega e^f - lambda_z sqrt(expo) sum pf_j s_j beta_j
  ## remove it. Signs are kept; a coefficient that would cross zero is set
  ## to zero and dropped (then the step is retried).
  if (polish) {
    pf <- dict$pen_factor
    for (pass in 1:3) {
      S <- which(beta != 0 | pf == 0); sg <- sign(beta[S]); sg[pf[S] == 0] <- 0
      th <- c(a, beta[S]); AD <- cbind(1, A_D[, S, drop = FALSE]); AB <- cbind(1, rep$tr$A[, S, drop = FALSE])
      pen <- c(0, lambda_z * sqrt(expo) * pf[S] * sg)
      obj <- function(t) sum(AD %*% t) - sum(rep$tr$omega * exp(AB %*% t)) - sum(pen * t)
      dropped <- FALSE
      for (it in 1:30) {
        mu <- rep$tr$omega * exp(as.vector(AB %*% th))
        g  <- colSums(AD) - colSums(AB * mu) - pen
        H  <- crossprod(AB, AB * mu)
        st <- solve(H, g); o0 <- obj(th); f <- 1
        while (obj(th + f * st) < o0 - 1e-10 && f > 1e-6) f <- f / 2
        th <- th + f * st
        if (max(abs(f * st)) < 1e-12) break
      }
      bS <- th[-1]; cross <- which(pf[S] > 0 & sign(bS) != sg)
      if (length(cross)) { beta[S[cross]] <- 0; dropped <- TRUE; next }
      a <- th[1]; beta[S] <- bS
      if (!dropped) break
    }
  }
  ## standardized scores at the solution (KKT diagnostics)
  mu_b <- rep$tr$omega * exp(a + as.vector(rep$tr$A %*% beta))
  z <- (colSums(A_D) - colSums(rep$tr$A * mu_b)) / sqrt(expo)
  pf <- dict$pen_factor; pen <- pf > 0
  act <- which(beta != 0)
  kkt_inactive <- max(c(0, abs(z[pen & beta == 0]) - lambda_z * pf[pen & beta == 0]))
  kkt_active   <- max(c(0, abs(abs(z[pen & beta != 0]) - lambda_z * pf[pen & beta != 0])))
  structure(list(
    beta = beta, a0 = a, lambda_z = lambda_z, lambda_glmnet = lam, expo = expo,
    active = act, active_smooth = intersect(act, dict$idx_smooth), active_bump = intersect(act, dict$idx_bump),
    z_at_fit = z, kkt_inactive = kkt_inactive, kkt_active = kkt_active,
    count_match = c(N_tr = nD, fitted = sum(mu_b)), npasses = fit$npasses
  ), class = "witness_fit_5d")
}

## Witness at atoms A: a + A beta
phi_5d <- function(fit, A) fit$a0 + as.vector(A %*% fit$beta)

print.witness_fit_5d <- function(x, ...) {
  cat(sprintf("witness fit: lambda_z %.3g; active smooth %d, bumps %d; KKT violation inactive %.2e, active %.2e; N_tr %d vs fitted %.1f\n",
              x$lambda_z, length(x$active_smooth), length(x$active_bump), x$kkt_inactive, x$kkt_active,
              x$count_match[["N_tr"]], x$count_match[["fitted"]]))
  invisible(x)
}

## -------------------------------------------------------------
## 2.5 Penalty pre-tuning on null toys (5-D analogue of choose_lambda)
## -------------------------------------------------------------
## For B null toys (no signal, nu* given, auxiliary constraint sigma_A):
## pushforward null fit on fold 1's training data, null representation,
## glmnet path over lambda_grid (z-units, same lambda for both penalized
## blocks, i.e. pf_smooth = 1 in the path), and the number of active smooth
## atoms and bumps at each lambda. Per block, the chosen lambda is the
## smallest on the grid at which the mean active fraction under H0 is at
## most alpha_active (the 1-D rule). The smooth block's penalty factor is
## then pf_smooth = lambda_smooth / lambda_bump (set it in make_dict_5d).
## The two blocks interact only weakly (smooth and bumps are orthogonal
## under f0). pf_path = "dict" runs the path with the dictionary's own
## penalty factors instead: the smooth block is then thresholded at
## lambda_z * pf_smooth, and at lambda_z = lambda_bump both blocks should
## show about alpha_active (the confirmation of the tuned configuration;
## the per-block lambdas it reports are then not meaningful for the smooth block).
choose_lambda_5d <- function(pools, dm, dict, B = 100L, nu_star = c(0, 0), sigma_A = CFG5$sigma_A,
                             N0 = CFG5$N0, K = CFG5$K, alpha_active = 0.05,
                             lambda_grid = exp(seq(log(6), log(0.5), length.out = 41)),
                             eps_bt = 1e-5, seed = 7101L, cores = 1L, pf_path = c("unit", "dict")) {
  pf_path <- match.arg(pf_path)
  lambda_grid <- sort(lambda_grid, decreasing = TRUE)
  is <- dict$idx_smooth; ib <- dict$idx_bump
  one <- function(b) {
    set.seed((seed * 7919 + b) %% .Machine$integer.max)
    t  <- add_delta_5d(draw_toy_5d(pools, N0 = N0, nu_star = nu_star, sigma_A = sigma_A, K = K), dm)
    iD <- which(t$fold != 1L); iR <- which(t$foldRpre != 1L)
    nf <- fit_null_push_5d(N = length(iD), gD = sum(t$dD[iD, "d1"]), Rpre = t$Rpre[iR, , drop = FALSE],
                           w = (K - 1) * N0 / (K * sum(t$pass0[iR])), nu_A = t$nu_A, sigma_A = t$sigma_A,
                           dm = dm, slope = FALSE)
    rep <- null_rep_5d(t, 1L, nf$nu, dict)
    A_D <- eval_atoms_5d(dict, t$xn[iD, , drop = FALSE], t$dD[iD, , drop = FALSE])
    X  <- rbind(rep$tr$A, A_D)
    y  <- c(rep(0, nrow(rep$tr$A)), rep(1 / eps_bt, nrow(A_D)))
    wt <- c(rep$tr$omega, rep(eps_bt, nrow(A_D)))
    pf <- dict$pen_factor
    if (pf_path == "unit") pf[is] <- ifelse(pf[is] > 0, 1, 0)   # common threshold for tuning
    lam <- lambda_grid * sqrt(rep$expo_tr) / sum(wt) * sum(pf) / length(pf)   # undo glmnet's pf rescaling
    fit <- glmnet::glmnet(X, y, family = "poisson", weights = wt, penalty.factor = pf, lambda = lam,
                          standardize = FALSE, intercept = TRUE, thresh = 1e-9, maxit = 1e5)
    Bm <- as.matrix(fit$beta) != 0
    rbind(smooth = colSums(Bm[is, , drop = FALSE]), bump = colSums(Bm[ib, , drop = FALSE]))
  }
  res <- if (cores > 1L) parallel::mclapply(seq_len(B), one, mc.cores = cores) else lapply(seq_len(B), one)
  bad <- vapply(res, inherits, logical(1), "try-error")
  if (any(bad)) stop("choose_lambda_5d: ", sum(bad), " toys failed; first: ", res[[which(bad)[1]]])
  S  <- sapply(res, function(r) r["smooth", ]); Bb <- sapply(res, function(r) r["bump", ])
  curve <- data.frame(lambda_z = lambda_grid,
                      frac_smooth = rowMeans(S) / length(is), frac_bump = rowMeans(Bb) / length(ib),
                      p_any_bump = rowMeans(Bb > 0))
  pick <- function(fr) { ok <- which(fr <= alpha_active); if (length(ok)) lambda_grid[max(ok)] else NA_real_ }
  ls <- pick(curve$frac_smooth); lb <- pick(curve$frac_bump)
  cat(sprintf("choose_lambda_5d: %d null toys, nu* = (%g, %g), sigma_A = (%g, %g): lambda_bump %.3f, lambda_smooth %.3f, pf_smooth %.3f\n",
              B, nu_star[1], nu_star[2], sigma_A[1], sigma_A[2], lb, ls, ls / lb))
  invisible(list(lambda_bump = lb, lambda_smooth = ls, pf_smooth = ls / lb, curve = curve,
                 alpha_active = alpha_active, B = B))
}

## -------------------------------------------------------------
## 2.6 Cross-fitted test for one toy (W4): the 5-D port of
##     utils.R::cvgof_fit_tilt + cvgof_test_quad
## -------------------------------------------------------------
## Per fold k (Theory_Methods.tex, two-sample extension):
##   S_k   = sum_{D_k} phi_k - sum_{test base} omega phi_k
##   g_k   = d/dnu E_nu[phi_k] at nu-hat^(-k): (E[phi], central difference in nu_s)
##   psi   influence direction of nu-hat: push = instrument (1, d1-hat);
##         ml = score u = (1, d1-hat + nu_s d2-hat)
##   J_k   push: Jacobian of the estimating equations (training-base C, G and their
##         nu_s central differences) + Sigma_A^-1; ml: information sum omega u u' + Sigma_A^-1
##   b_k   = (K - 1) J_k^-T g_k;  phi~_k = phi_k - b_k' psi;  c_aux += J_k^-T g_k
##   V^D_k = sum_{test base} omega phi~_k^2          (Poisson: uncentered)
##   V^R_k = Var of the fixed-size reference sum of omega phi~_k over the test
##           pre-selection reference, stratified by pass/fail at nu = 0
##   v_aux = c_aux' Sigma_A^-1 c_aux
## Fitting randomness (degenerate regime under H0), as in cvgof_test_quad:
##   V_cross = sum_{j != k} E_j E_k tr(D_k H_k^-1 Mt_kj D_j H_j^-1 Mt_jk),
##   on a common measure (the full-data fitted null, fold 0), H_k the fold-k
##   Hessian on its active design [1 | tangent | active penalized], D_k the
##   damping: intercept 1 - rho_n, T1 1 - rho_s, T2 1 (not a null direction),
##   selected penalized atoms 1 ("none": slope-one influence on the active set;
##   the unified rule, validated in 1-D and 5-D). Also reported: "soft"
##   (|beta| sqrt(expo) / (|beta| sqrt(expo) + lambda pf), the retired 1-D
##   heuristic) and "block" (smooth 1, bumps soft).
##   Mt_kj = M_kj (data noise only, the 1-D form) or M_kj + r C_kj with C the
##   fitted-null covariance of the atoms and r = N0 / N_R (Conjecture ts-cov:
##   the Reference contributes to the contrast). Both are reported.
## Headline: Z_fit = S / sqrt(sum V^D + sum V^R + v_aux + V_cross(none, with r C)).
## fixed_beta (a coefficient vector over the dictionary columns, plus a0 in
## attr "a0") replaces the fit by a known witness: the fixed-witness check.
cvgof_5d <- function(toy, dm, dict, lambda_z, method = c("push", "ml"), eps = 1e-3,
                     fixed_beta = NULL, null_fits = NULL, alpha = 0.05) {
  method <- match.arg(method); .check_dm(dm, "cvgof_5d")
  if (is.null(toy$dD)) toy <- add_delta_5d(toy, dm)
  K <- toy$K; N0 <- toy$N0
  nf <- null_fits %||% (if (method == "push") null_fits_push_5d(toy, dm) else null_fits_5d(toy))
  nu_of <- function(k) c(nf$nu_n[nf$fold == k], nf$nu_s[nf$fold == k])
  sA <- toy$sigma_A; pA <- ifelse(is.finite(sA), 1 / sA^2, 0); SigAinv <- diag(pA, 2L)
  ## common measure for the cross-fold covariance: full-data fitted null
  r0 <- null_rep_5d(toy, 0L, nu_of(0L), dict, method, parts = "tr")
  A_ref <- r0$tr$A; wp_ref <- r0$tr$omega / sum(r0$tr$omega)
  mean_ref <- colSums(A_ref * wp_ref)
  psi_of <- function(D, nu_s) if (method == "push") cbind(1, D[, "d1"]) else cbind(1, D[, "d1"] + nu_s * D[, "d2"])

  S <- S_or <- VD <- VD_raw <- VR <- VR_raw <- VX <- numeric(K)
  g <- b <- matrix(NA_real_, K, 2L); c_aux <- numeric(2L); recs <- vector("list", K)
  for (k in seq_len(K)) {
    rep <- null_rep_5d(toy, k, nu_of(k), dict, method, eps = eps)
    iD <- which(toy$fold != k); iE <- which(toy$fold == k)
    if (is.null(fixed_beta)) {
      fit <- fit_witness_5d(rep, eval_atoms_5d(dict, toy$xn[iD, , drop = FALSE], toy$dD[iD, , drop = FALSE]),
                            lambda_z, dict)
    } else {
      fit <- list(beta = as.numeric(fixed_beta), a0 = attr(fixed_beta, "a0") %||% 0,
                  active_smooth = integer(0), active_bump = integer(0))
    }
    A_E <- eval_atoms_5d(dict, toy$xn[iE, , drop = FALSE], toy$dD[iE, , drop = FALSE])
    ph  <- function(A) fit$a0 + as.vector(A %*% fit$beta)
    phx <- ph(A_E); phb <- ph(rep$te$A); om <- rep$te$omega
    S[k] <- sum(phx) - sum(om * phb)
    E <- function(B) sum(B$omega * ph(B$A))
    g[k, ] <- c(sum(om * phb), (E(rep$te_p) - E(rep$te_m)) / (2 * eps))
    nus <- rep$nu[["nu_s"]]
    psx <- psi_of(toy$dD[iE, , drop = FALSE], nus); psb <- psi_of(rep$te$D, nus)
    if (method == "push") {
      CG <- rep$CG
      J0 <- rbind(c(CG["c", "C"], (CG["p", "C"] - CG["m", "C"]) / (2 * eps)),
                  c(CG["c", "G"], (CG["p", "G"] - CG["m", "G"]) / (2 * eps)))
    } else {
      pst <- psi_of(rep$tr$D, nus); J0 <- crossprod(pst, pst * rep$tr$omega)
    }
    J <- J0 + SigAinv
    Jtg <- solve(t(J), g[k, ]); b[k, ] <- (K - 1) * Jtg; c_aux <- c_aux + Jtg
    ptx <- phx - as.vector(psx %*% b[k, ]); ptb <- phb - as.vector(psb %*% b[k, ])
    S_or[k] <- sum(ptx) - sum(om * ptb)
    VD[k] <- sum(om * ptb^2); VD_raw[k] <- sum(om * phb^2); VX[k] <- sum(ptx^2)
    ## reference variance: fixed-size sums over the test pre-selection reference,
    ## stratified by pass/fail at nu = 0 (fixed counts per stratum)
    vR <- function(vals) {
      v <- numeric(rep$M_te); v[rep$te$src] <- om * vals
      sum(vapply(split(v, rep$te_pass0), function(x) length(x) * mean((x - mean(x))^2), numeric(1)))
    }
    VR[k] <- vR(ptb); VR_raw[k] <- vR(phb)
    ## fitting-randomness record on the common measure
    if (is.null(fixed_beta)) {
      ia  <- c(dict$idx_tan, fit$active_smooth, fit$active_bump)
      A_act <- cbind(1, A_ref[, ia, drop = FALSE])
      mu_ref <- rep$expo_tr * wp_ref * exp(fit$a0 + as.vector(A_ref %*% fit$beta))
      I_s <- J0[2, 2]; rho_s <- I_s / (I_s + pA[2]); rho_n <- rep$expo_tr / (rep$expo_tr + pA[1])
      ipen <- c(fit$active_smooth, fit$active_bump)
      bb <- abs(fit$beta[ipen]) * sqrt(rep$expo_tr)
      damp <- c(1 - rho_n, 1 - rho_s, 1, bb / (bb + lambda_z * dict$pen_factor[ipen]))
      damp_none <- c(1 - rho_n, 1 - rho_s, 1, rep(1, length(ipen)))   # selected atoms treated as free
      ## "block": the 1-D treatment per block -- smooth atoms count fully (as when
      ## unpenalized in 1-D), bumps soft-damped (the 1-D bump rule)
      is_sm <- ipen %in% dict$idx_smooth
      damp_block <- c(1 - rho_n, 1 - rho_s, 1, ifelse(is_sm, 1, bb / (bb + lambda_z * dict$pen_factor[ipen])))
      recs[[k]] <- list(A_act = A_act, mu_ref = mu_ref, damp = damp, damp_none = damp_none, damp_block = damp_block,
                        expo_te = rep$expo_te, mean_act = c(1, mean_ref[ia]))
    }
    recs[[k]]$diag <- c(nu_n = rep$nu[["nu_n"]], nu_s = nus, n_smooth = length(fit$active_smooth),
                        n_bump = length(fit$active_bump), expo_tr = rep$expo_tr, expo_te = rep$expo_te)
  }
  v_aux <- as.numeric(crossprod(c_aux, SigAinv %*% c_aux))

  ## cross-fold covariance, data-only Grams (1-D form) and with the Reference term,
  ## for three damping rules of the selected penalized atoms: "soft" (soft-threshold
  ## factor |beta| sqrt(E) / (|beta| sqrt(E) + lambda pf) for smooth and bumps),
  ## "none" (selected atoms treated as unpenalized: damping 1), and "block" (the
  ## 1-D treatment per block: smooth damping 1, bumps soft)
  Vc <- c(data = 0, ref = 0, data_nd = 0, ref_nd = 0, data_bl = 0, ref_bl = 0)
  if (is.null(fixed_beta)) {
    r_ratio <- N0 / toy$NR
    Hi0  <- lapply(recs, function(f) solve(crossprod(f$A_act, f$A_act * f$mu_ref)))
    Hinv <- Map(function(H, f) f$damp * H, Hi0, recs)
    Hnd  <- Map(function(H, f) f$damp_none * H, Hi0, recs)
    Hbl  <- Map(function(H, f) f$damp_block * H, Hi0, recs)
    for (j in seq_len(K - 1L)) for (k in (j + 1L):K) {
      M_kj <- crossprod(recs[[k]]$A_act, recs[[j]]$A_act * wp_ref)
      C_kj <- M_kj - tcrossprod(recs[[k]]$mean_act, recs[[j]]$mean_act)
      for (nm in names(Vc)) {
        Mt <- if (grepl("^data", nm)) M_kj else M_kj + r_ratio * C_kj
        Hl <- if (grepl("_nd$", nm)) Hnd else if (grepl("_bl$", nm)) Hbl else Hinv
        Vc[nm] <- Vc[nm] + 2 * recs[[j]]$expo_te * recs[[k]]$expo_te *
          sum((Hl[[k]] %*% Mt) * t(Hl[[j]] %*% t(Mt)))
      }
    }
  }
  St <- sum(S)
  Z <- function(v) St / sqrt(v)
  res <- c(S = St,
           ## headline: damping rule "none" with the Reference term (unified rule)
           Z_fit   = Z(sum(VD) + sum(VR) + v_aux + Vc[["ref_nd"]]),
           Z_fit_D = Z(sum(VD) + sum(VR) + v_aux + Vc[["data_nd"]]),
           Z_fit_soft  = Z(sum(VD) + sum(VR) + v_aux + Vc[["ref"]]),
           Z_fit_Dsoft = Z(sum(VD) + sum(VR) + v_aux + Vc[["data"]]),
           Z_fit_bl  = Z(sum(VD) + sum(VR) + v_aux + Vc[["ref_bl"]]),
           Z_fit_Dbl = Z(sum(VD) + sum(VR) + v_aux + Vc[["data_bl"]]),
           Z_orth  = Z(sum(VD) + sum(VR) + v_aux),
           Z_orthD = Z(sum(VD) + v_aux),                 # no Reference variance (r = 0 view)
           Z_raw   = Z(sum(VD_raw) + sum(VR_raw)),
           V_D = sum(VD), V_R = sum(VR), v_aux = v_aux,
           V_cross_data = Vc[["data_nd"]], V_cross_ref = Vc[["ref_nd"]],          # "none" (headline)
           V_cross_data_soft = Vc[["data"]], V_cross_ref_soft = Vc[["ref"]],
           V_cross_data_bl = Vc[["data_bl"]], V_cross_ref_bl = Vc[["ref_bl"]],
           V_X = sum(VX), S_orth_num = sum(S_or))
  dg <- do.call(rbind, lapply(recs, `[[`, "diag"))
  structure(list(stat = res, p_value = stats::pnorm(res[["Z_fit"]], lower.tail = FALSE),
                 reject = stats::pnorm(res[["Z_fit"]], lower.tail = FALSE) < alpha,
                 S_k = S, g = g, b = b, per_fold = dg, method = method, lambda_z = lambda_z,
                 n_sig = toy$n_sig, nu_star = toy$nu_star), class = "cvgof_5d")
}


print.cvgof_5d <- function(x, ...) {
  s <- x$stat
  cat(sprintf("cvgof_5d (%s): S = %.2f; Z_fit %.2f (p %.3g) | Z_fit_D %.2f | Z_orth %.2f | Z_raw %.2f\n",
              x$method, s[["S"]], s[["Z_fit"]], x$p_value, s[["Z_fit_D"]], s[["Z_orth"]], s[["Z_raw"]]))
  cat(sprintf("  V_D %.2f  V_R %.2f  v_aux %.2f  V_cross data %.2f / with ref %.2f;  V_X/V_D %.3f;  active per fold: smooth %.1f, bumps %.1f\n",
              s[["V_D"]], s[["V_R"]], s[["v_aux"]], s[["V_cross_data"]], s[["V_cross_ref"]], s[["V_X"]] / s[["V_D"]],
              mean(x$per_fold[, "n_smooth"]), mean(x$per_fold[, "n_bump"])))
  invisible(x)
}

## -------------------------------------------------------------
## 2.7 Validation runners (W5)
## -------------------------------------------------------------
## Fixed-witness check: a KNOWN witness phi = a0 + A beta (no learning),
## B toys at nu*; var(S) against the mean studentizer V_D + V_R + v_aux,
## separately for each method. Holding phi fixed switches the fitting
## randomness off, so this tests the plug-in correction, v_aux and the
## Reference variance, not V_cross.
.default_fixed_beta_5d <- function(dict) {
  b <- numeric(dict$p_tan + dict$p_smooth + dict$J)
  b[dict$idx_tan[1]] <- 0.05                  # along the nuisance: exercises the plug-in term
  b[dict$idx_smooth[1:3]] <- c(0.05, -0.03, 0.02)
  b[dict$idx_bump[order(-dict$bump_mass)[1:2]]] <- c(0.08, -0.05)
  attr(b, "a0") <- 0.02                       # count channel
  b
}

## Null (or signal) study: B toys, each analysed by every method in `methods`
## on the same data. Returns per-toy stats and prints a summary.
## progress: optional file; each finished toy appends one line
##   "HH:MM:SS <tag> toy <b>/<B> <secs>s"  (safe from parallel workers),
## and the study writes a start and an end line. Follow it with
##   tail -f <file>      or      grep -c "<tag> toy" <file>
run_study_5d <- function(pools, dm, dict, lambda_z, B = 200L, methods = c("push", "ml"),
                         nu_star = c(0, 0), sigma_A = CFG5$sigma_A, signal = NULL, NS = 0,
                         N0 = CFG5$N0, K = CFG5$K, fixed_beta = NULL, seed = 1L, cores = 1L,
                         progress = NULL, tag = "study") {
  logp <- function(msg) if (!is.null(progress))
    cat(format(Sys.time(), "%H:%M:%S"), tag, msg, "\n", file = path.expand(progress), append = TRUE)
  logp(sprintf("start: B = %d, methods %s, nu* = (%g, %g), %s, cores %d", B, paste(methods, collapse = "+"),
               nu_star[1], nu_star[2], if (is.null(fixed_beta)) "learned witness" else "fixed witness", cores))
  t_start <- proc.time()[["elapsed"]]
  one <- function(b) {
    t_toy <- proc.time()[["elapsed"]]
    on.exit(logp(sprintf("toy %d/%d %.0fs", b, B, proc.time()[["elapsed"]] - t_toy)), add = TRUE)
    set.seed((seed * 100003 + b) %% .Machine$integer.max)
    t <- add_delta_5d(draw_toy_5d(pools, N0 = N0, signal = signal, NS = NS, nu_star = nu_star,
                                  sigma_A = sigma_A, K = K), dm)
    do.call(rbind, lapply(methods, function(m) {
      t0 <- proc.time()[["elapsed"]]
      r <- cvgof_5d(t, dm, dict, lambda_z, method = m, fixed_beta = fixed_beta)
      data.frame(toy = b, method = m, t(r$stat), n_sig = t$n_sig,
                 nu_s_hat = mean(r$per_fold[, "nu_s"]), smooth = mean(r$per_fold[, "n_smooth"]),
                 bumps = mean(r$per_fold[, "n_bump"]), secs = proc.time()[["elapsed"]] - t0)
    }))
  }
  res <- if (cores > 1L) parallel::mclapply(seq_len(B), one, mc.cores = cores) else lapply(seq_len(B), one)
  bad <- vapply(res, inherits, logical(1), "try-error")
  if (any(bad)) warning("run_study_5d: ", sum(bad), " toys failed; first: ", res[[which(bad)[1]]])
  R <- do.call(rbind, res[!bad])
  logp(sprintf("end: %d/%d toys ok, %.1f min", sum(!bad), B, (proc.time()[["elapsed"]] - t_start) / 60))
  attr(R, "meta") <- list(nu_star = nu_star, sigma_A = sigma_A, signal = signal, NS = NS, N0 = N0, K = K,
                          lambda_z = lambda_z, B = B, fixed = !is.null(fixed_beta), seed = seed)
  summary_study_5d(R)
  invisible(R)
}

summary_study_5d <- function(R) {
  m <- attr(R, "meta")
  cat(sprintf("study: %d toys, nu* = (%g, %g), sigma_A = (%g, %g), signal %s (NS %g), lambda_z %g%s\n",
              length(unique(R$toy)), m$nu_star[1], m$nu_star[2], m$sigma_A[1], m$sigma_A[2],
              if (is.null(m$signal)) "none" else m$signal, m$NS, m$lambda_z, if (m$fixed) ", FIXED witness" else ""))
  zc <- c("Z_fit", "Z_fit_D", "Z_orth", "Z_raw")
  for (mm in unique(R$method)) {
    d <- R[R$method == mm, ]
    cat(sprintf("  [%s]  var(S)/mean(V_D+V_R+v_aux) = %.3f;  mean S = %.2f (se %.2f);  V_R/V_D = %.3f;  %.1fs/toy\n",
                mm, stats::var(d$S) / mean(d$V_D + d$V_R + d$v_aux), mean(d$S), stats::sd(d$S) / sqrt(nrow(d)),
                mean(d$V_R / d$V_D), mean(d$secs)))
    if (!m$fixed) {
      base <- d$V_D + d$V_R + d$v_aux
      rt <- function(v) if (is.null(v)) NA_real_ else stats::var(d$S) / mean(base + v)
      if (!is.null(d$V_cross_ref_nd)) {      # results saved before the "none" default (Z_fit was soft)
        d$V_cross_ref_soft <- d$V_cross_ref; d$V_cross_data_soft <- d$V_cross_data
        d$V_cross_ref <- d$V_cross_ref_nd;   d$V_cross_data <- d$V_cross_data_nd
      }
      cat(sprintf("        var(S)/mean(... + V_cross)  [ref | data]:  none (headline) %.3f | %.3f;  block %.3f | %.3f;  soft %.3f | %.3f\n",
                  rt(d$V_cross_ref), rt(d$V_cross_data), rt(d$V_cross_ref_bl), rt(d$V_cross_data_bl),
                  rt(d$V_cross_ref_soft), rt(d$V_cross_data_soft)))
      cat(sprintf("        active per fold: smooth %.2f, bumps %.2f;  cor(S, V_D) %.2f\n",
                  mean(d$smooth), mean(d$bumps), stats::cor(d$S, d$V_D)))
    }
    zc2 <- intersect(c(zc, "Z_fit_bl", "Z_fit_Dbl", "Z_fit_soft", "Z_fit_Dsoft", "Z_fit_nd", "Z_fit_Dnd"), names(d))
    if (m$fixed) zc2 <- zc
    tab <- t(sapply(zc2, function(z) { x <- d[[z]]
      c(mean = mean(x), sd = stats::sd(x), `P(Z>1.645)` = mean(x > 1.645), `P(Z>2.326)` = mean(x > 2.326)) }))
    print(round(tab, 3))
  }
  invisible(R)
}
