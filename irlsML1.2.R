# =========================
# VarGuid ML (Algorithm 1)
# =========================

# Helpers --------------------------------------------------------------

.rmse <- function(y, yhat) sqrt(mean((y - yhat)^2, na.rm = TRUE))

.make_matrix <- function(x) {
  if (is.data.frame(x)) {
    # Convert factors safely
    mm <- model.matrix(~ . - 1, data = x)
    return(mm)
  }
  as.matrix(x)
}

.scale_fit <- function(X) {
  mu <- colMeans(X)
  sd <- apply(X, 2, sd)
  sd[sd == 0] <- 1
  list(mu = mu, sd = sd)
}
.scale_apply <- function(X, scaler) {
  sweep(sweep(X, 2, scaler$mu, "-"), 2, scaler$sd, "/")
}

.kfold_ids <- function(n, k, seed = 1) {
  set.seed(seed)
  fold <- sample(rep(seq_len(k), length.out = n))
  fold
}

# --- ML backends: train/predict for f and g ---------------------------

.train_rfsrc <- function(X, y, w = NULL, ...) {
  if (!requireNamespace("randomForestSRC", quietly = TRUE)) {
    stop("Package 'randomForestSRC' is required for method='rfsrc'.")
  }
  dat <- data.frame(y = y, X)
  fit <- randomForestSRC::rfsrc(
    y ~ ., data = dat,
    case.wt = w,
    importance = TRUE,
    ...
  )
  fit
}
.predict_rfsrc <- function(fit, Xnew) {
  dat <- data.frame(Xnew)
  as.numeric(randomForestSRC::predict.rfsrc(fit, newdata = dat)$predicted)
}
.vimp_rfsrc <- function(fit) {
  # rfsrc stores importance, but vimp() is the usual path
  if (!requireNamespace("randomForestSRC", quietly = TRUE)) return(NULL)
  vi <- try(randomForestSRC::vimp(fit)$importance, silent = TRUE)
  if (inherits(vi, "try-error")) return(NULL)
  vi
}

.train_bart <- function(X, y, w = NULL, seed = 1, ndpost = 1000, nskip = 200, ...) {
  if (!requireNamespace("dbarts", quietly = TRUE)) {
    stop("Package 'dbarts' is required for method='bart'.")
  }
  set.seed(seed)
  
  fit <- dbarts::bart(
    x.train = X,
    y.train = as.numeric(y),
    nskip = nskip,
    ndpost = ndpost,
    weights = w,          # <-- real weights supported in your version
    keeptrees = TRUE,     # <-- correct arg name
    verbose = FALSE,
    ...
  )
  fit
}


.predict_bart <- function(fit, Xnew) {
  p <- dbarts:::predict.bart(fit, newdata = Xnew)  # force dbarts predict
  as.numeric(colMeans(p))
}


.vimp_bart <- function(fit, colnames_X = NULL) {
  # crude variable usage counts (not permutation VIMP)
  vc <- try(dbarts::varcount(fit), silent = TRUE)
  if (inherits(vc, "try-error")) return(NULL)
  vc <- colMeans(vc)
  if (!is.null(colnames_X) && length(vc) == length(colnames_X)) {
    names(vc) <- colnames_X
  }
  sort(vc, decreasing = TRUE)
}



.train_torch_mlp <- function(X, y, w = NULL,
                             spec = list(hidden = c(64, 64),
                                         dropout = 0, lr = 1e-3,
                                         epochs = 100, batch_size = 64,
                                         weight_decay = 0,
                                         patience = 0,
                                         eval_freq = 1),
                             seed = 1) {
  for (pkg in c("mlr3", "mlr3torch")) {
    if (!requireNamespace(pkg, quietly = TRUE)) stop("Need package: ", pkg)
  }
  
  X <- as.matrix(X)
  dat <- as.data.frame(X)
  dat$y <- as.numeric(y)
  
  task <- mlr3::TaskRegr$new(id = "varguid_torch_train", backend = dat, target = "y")
  
  lrn <- mlr3::lrn("regr.mlp")
  ids <- lrn$param_set$ids()
  
  # Map spec -> learner params
  hidden <- as.integer(spec$hidden %||% c(64, 64))
  neurons <- as.integer(hidden[1])
  n_layers <- as.integer(length(hidden))
  
  if ("neurons" %in% ids) lrn$param_set$values$neurons <- neurons
  if ("n_layers" %in% ids) lrn$param_set$values$n_layers <- n_layers
  if ("p" %in% ids) lrn$param_set$values$p <- as.numeric(spec$dropout %||% 0)
  
  if ("opt.lr" %in% ids) lrn$param_set$values$`opt.lr` <- as.numeric(spec$lr %||% 1e-3)
  if ("opt.weight_decay" %in% ids) lrn$param_set$values$`opt.weight_decay` <- as.numeric(spec$weight_decay %||% 0)
  
  if ("epochs" %in% ids) lrn$param_set$values$epochs <- as.integer(spec$epochs %||% 100)
  if ("batch_size" %in% ids) lrn$param_set$values$batch_size <- as.integer(spec$batch_size %||% 64)
  
  # Keep these consistent with the tuning workaround:
  # measures_* required in your version; keep them empty to avoid validation machinery.
  if ("measures_train" %in% ids) lrn$param_set$values$measures_train <- list()
  if ("measures_valid" %in% ids) lrn$param_set$values$measures_valid <- list()
  
  # Disable early stopping/validation
  if ("patience" %in% ids) lrn$param_set$values$patience <- 0L
  if ("eval_freq" %in% ids) lrn$param_set$values$eval_freq <- 1  # must be >= 0.5 in your setup
  if ("min_delta" %in% ids) lrn$param_set$values$min_delta <- 0
  
  # Seed/device
  if ("seed" %in% ids) lrn$param_set$values$seed <- as.integer(seed)
  if ("device" %in% ids) lrn$param_set$values$device <- "cpu"
  if ("shuffle" %in% ids) lrn$param_set$values$shuffle <- TRUE
  
  # Case weights: mlr3 uses "weights" argument to train() at task level for some learners.
  # regr.mlp may or may not support weights. We'll approximate via resampling if weights provided.
  if (!is.null(w)) {
    w <- as.numeric(w)
    w[!is.finite(w)] <- 0
    w <- pmax(w, 0)
    if (sum(w) == 0) w <- rep(1, length(w))
    prob <- w / sum(w)
    idx <- sample(seq_len(nrow(dat)), size = nrow(dat), replace = TRUE, prob = prob)
    dat2 <- dat[idx, , drop = FALSE]
    task2 <- mlr3::TaskRegr$new(id = "varguid_torch_train_w", backend = dat2, target = "y")
    lrn$train(task2)
    return(list(learner = lrn, colnames = colnames(X)))
  }
  
  lrn$train(task)
  list(learner = lrn, colnames = colnames(X))
}

.predict_torch_mlp <- function(fit, Xnew) {
  for (pkg in c("mlr3")) {
    if (!requireNamespace(pkg, quietly = TRUE)) stop("Need package: ", pkg)
  }
  
  Xnew <- as.matrix(Xnew)
  dat <- as.data.frame(Xnew)
  
  # mlr3 prediction requires a Task; easiest is to create a dummy target
  dat$y <- 0
  task_new <- mlr3::TaskRegr$new(id = "varguid_torch_pred", backend = dat, target = "y")
  
  pred <- fit$learner$predict(task_new)
  as.numeric(pred$response)
}

`%||%` <- function(a, b) if (!is.null(a)) a else b

# Permutation importance (generic; usable for any method) --------------
.permutation_vimp <- function(predict_fun, fitted_obj, X, y, nrep = 3, seed = 1) {
  set.seed(seed)
  base <- .rmse(y, predict_fun(fitted_obj, X))
  p <- ncol(X)
  out <- numeric(p)
  for (j in seq_len(p)) {
    inc <- numeric(nrep)
    for (r in seq_len(nrep)) {
      Xp <- X
      Xp[, j] <- sample(Xp[, j])
      inc[r] <- .rmse(y, predict_fun(fitted_obj, Xp)) - base
    }
    out[j] <- mean(inc)
  }
  names(out) <- colnames(X)
  sort(out, decreasing = TRUE)
}

# Cross-validated RMSE for validation check ---------------------------
# =========================
# Updated CV RMSE
# Supports optional cross-fitting + (optional) weight stabilization
# =========================
.cv_rmse <- function(method, X, y,
                     w = NULL,
                     k = 5,
                     seed = 1,
                     torch_spec = NULL,
                     # weight stabilization (optional for CV)
                     stabilize_weights = TRUE,
                     alpha = 1.0,
                     w_clip_q = c(0.05, 0.95),
                     w_min = NULL,
                     w_max = NULL,
                     ...) {
  fold <- .kfold_ids(nrow(X), k = k, seed = seed)
  rmses <- numeric(k)
  
  # helper: stabilize weights
  .stab_w <- function(wvec) {
    if (is.null(wvec)) return(NULL)
    wv <- as.numeric(wvec)
    wv[!is.finite(wv)] <- NA_real_
    if (all(is.na(wv))) return(NULL)
    
    # tempering: w <- w^alpha  (alpha<1 dampens extremes)
    alpha_use <- as.numeric(alpha)
    if (!is.finite(alpha_use) || alpha_use <= 0) alpha_use <- 1.0
    wv <- wv^alpha_use
    
    # clip by quantiles unless explicit bounds provided
    if (is.null(w_min) || is.null(w_max)) {
      qq <- stats::quantile(wv, probs = w_clip_q, na.rm = TRUE, names = FALSE, type = 7)
      lo <- qq[1]; hi <- qq[2]
    } else {
      lo <- as.numeric(w_min); hi <- as.numeric(w_max)
    }
    if (!is.finite(lo)) lo <- min(wv, na.rm = TRUE)
    if (!is.finite(hi)) hi <- max(wv, na.rm = TRUE)
    if (hi < lo) { tmp <- lo; lo <- hi; hi <- tmp }
    
    wv <- pmin(pmax(wv, lo), hi)
    wv[is.na(wv)] <- stats::median(wv, na.rm = TRUE)
    wv
  }
  
  for (kk in seq_len(k)) {
    tr <- fold != kk
    te <- fold == kk
    
    Xtr <- X[tr, , drop = FALSE]
    ytr <- y[tr]
    wtr <- if (is.null(w)) NULL else w[tr]
    
    Xte <- X[te, , drop = FALSE]
    yte <- y[te]
    
    if (!is.null(wtr) && stabilize_weights) {
      wtr <- .stab_w(wtr)
    }
    
    if (method == "rfsrc") {
      fit <- .train_rfsrc(Xtr, ytr, w = wtr, ...)
      pred <- .predict_rfsrc(fit, Xte)
      
    } else if (method == "bart") {
      fit <- .train_bart(Xtr, ytr, w = wtr, seed = seed + kk, ...)
      pred <- .predict_bart(fit, Xte)
      
    } else if (method == "torch") {
      spec <- torch_spec %||% list()
      fit <- .train_torch_mlp(Xtr, ytr, w = wtr, spec = spec, seed = seed + kk)
      pred <- .predict_torch_mlp(fit, Xte)
      
    } else {
      stop("Unknown method in .cv_rmse().")
    }
    
    rmses[kk] <- .rmse(yte, pred)
  }
  
  mean(rmses)
}

# =========================
# Main fit function
# =========================
# =========================
# Updated VarGuid fit
# Implements:
# 1) weight stabilization (tempering + clipping)
# 3) delayed weighting (warmup iterations)
# 5) margin rule for accepting weighted fit
# 7) cross-fitting for g (default ON; can turn off for speed)
# fit_fast <- varguid_fit_ml(
#   x = Xtr, y = ytr,
#   method = "rfsrc",
#   T = 10,
#   warmup = 0,
#   use_margin = FALSE,
#   crossfit_g = FALSE,
#   k_cv = 0
# )
# =========================
varguid_fit_ml <- function(x, y,
                           method = c("rfsrc", "bart", "torch"),
                           # baseline + varguid controls
                           c_init = NULL,
                           eps = 1e-6,
                           tau = 1e-6,
                           T = 25,
                           k_cv = 5,
                           seed = 1,
                           torch_spec = NULL,
                           
                           # ---- (1) weight stabilization ----
                           stabilize_weights = TRUE,
                           alpha = 0.5,                 # default tempering (more stable than 1)
                           w_clip_q = c(0.05, 0.95),     # winsorize weights by default
                           w_min = NULL,
                           w_max = NULL,
                           
                           # ---- (3) delayed weighting ----
                           warmup = 1L,                  # default: 1 iteration unweighted
                           # if warmup >= T, it becomes baseline-only iterations
                           
                           # ---- (5) margin rule ----
                           use_margin = TRUE,
                           margin_type = c("relative", "absolute"),
                           margin = 0.005,               # default: 0.5% RMSE improvement needed
                           # weighted accepted only if rmse_w < rmse_u - threshold
                           
                           # ---- (7) cross-fitting for g ----
                           crossfit_g = TRUE,
                           g_folds = 5,
                           
                           # importance
                           importance = c("auto", "permutation", "none"),
                           vimp_nrep = 3,
                           ...) {
  
  method <- match.arg(method)
  importance <- match.arg(importance)
  margin_type <- match.arg(margin_type)
  
  X <- .make_matrix(x)
  y <- as.numeric(y)
  n <- nrow(X)
  
  # ----------------------------
  # A) Baseline (unweighted) fit
  # ----------------------------
  if (method == "rfsrc") {
    base_f_fit <- .train_rfsrc(X, y, w = NULL, ...)
    base_f_hat <- .predict_rfsrc(base_f_fit, X)
  } else if (method == "bart") {
    base_f_fit <- .train_bart(X, y, w = NULL, seed = seed + 1, ...)
    base_f_hat <- .predict_bart(base_f_fit, X)
  } else if (method == "torch") {
    spec <- torch_spec %||% list()
    base_f_fit <- .train_torch_mlp(X, y, w = NULL, spec = spec, seed = seed + 1)
    base_f_hat <- .predict_torch_mlp(base_f_fit, X)
  } else stop("Unknown method.")
  
  # --------------------------------
  # B) VarGuid initialization: g^(0)
  # --------------------------------
  if (is.null(c_init)) c_init <- var(y, na.rm = TRUE)
  g <- rep(max(c_init, eps), n)
  
  f_fit <- NULL
  g_fit <- NULL
  f_pred_prev <- as.numeric(base_f_hat)
  history <- list()
  
  warmup <- as.integer(warmup)
  warmup <- max(0L, warmup)
  
  # helper: stabilize weights (1)
  .stab_w <- function(wvec) {
    if (is.null(wvec)) return(NULL)
    wv <- as.numeric(wvec)
    wv[!is.finite(wv)] <- NA_real_
    if (all(is.na(wv))) return(NULL)
    
    # tempering
    alpha_use <- as.numeric(alpha)
    if (!is.finite(alpha_use) || alpha_use <= 0) alpha_use <- 1.0
    wv <- wv^alpha_use
    
    # clip
    if (is.null(w_min) || is.null(w_max)) {
      qq <- stats::quantile(wv, probs = w_clip_q, na.rm = TRUE, names = FALSE, type = 7)
      lo <- qq[1]; hi <- qq[2]
    } else {
      lo <- as.numeric(w_min); hi <- as.numeric(w_max)
    }
    if (!is.finite(lo)) lo <- min(wv, na.rm = TRUE)
    if (!is.finite(hi)) hi <- max(wv, na.rm = TRUE)
    if (hi < lo) { tmp <- lo; lo <- hi; hi <- tmp }
    
    wv <- pmin(pmax(wv, lo), hi)
    wv[is.na(wv)] <- stats::median(wv, na.rm = TRUE)
    wv
  }
  
  # helper: margin threshold (5)
  .margin_thresh <- function(rmse_u) {
    if (!use_margin) return(0)
    m <- as.numeric(margin)
    if (!is.finite(m) || m < 0) m <- 0
    if (margin_type == "relative") m * as.numeric(rmse_u) else m
  }
  
  # helper: fit/predict f given weights
  .fit_f <- function(wfit, seed_shift = 0L) {
    if (method == "rfsrc") {
      fit <- .train_rfsrc(X, y, w = wfit, ...)
      pred <- .predict_rfsrc(fit, X)
    } else if (method == "bart") {
      fit <- .train_bart(X, y, w = wfit, seed = seed + seed_shift, ...)
      pred <- .predict_bart(fit, X)
    } else if (method == "torch") {
      spec <- torch_spec %||% list()
      fit <- .train_torch_mlp(X, y, w = wfit, spec = spec, seed = seed + seed_shift)
      pred <- .predict_torch_mlp(fit, X)
    } else stop("Unknown method.")
    list(fit = fit, pred = as.numeric(pred))
  }
  
  # helper: cross-fit g on r^2 (7)
  .crossfit_g_pred <- function(r2, seed_shift = 0L) {
    foldg <- .kfold_ids(n, k = g_folds, seed = seed + 7000 + seed_shift)
    g_hat <- rep(NA_real_, n)
    
    for (kk in seq_len(g_folds)) {
      tr <- foldg != kk
      te <- foldg == kk
      
      Xtr <- X[tr, , drop = FALSE]
      r2tr <- r2[tr]
      Xte <- X[te, , drop = FALSE]
      
      if (method == "rfsrc") {
        fit <- .train_rfsrc(Xtr, r2tr, w = NULL, ...)
        g_hat[te] <- .predict_rfsrc(fit, Xte)
        
      } else if (method == "bart") {
        fit <- .train_bart(Xtr, r2tr, w = NULL, seed = seed + 8000 + seed_shift + kk, ...)
        g_hat[te] <- .predict_bart(fit, Xte)
        
      } else if (method == "torch") {
        spec <- torch_spec %||% list()
        fit <- .train_torch_mlp(Xtr, r2tr, w = NULL, spec = spec, seed = seed + 9000 + seed_shift + kk)
        g_hat[te] <- .predict_torch_mlp(fit, Xte)
      }
    }
    
    g_hat <- pmax(as.numeric(g_hat), eps)
    g_hat
  }
  
  # -------------------------
  # C) VarGuid iterations
  # -------------------------
  for (t in seq_len(T)) {
    
    # (3) delayed weighting: during warmup, force unweighted
    if (t <= warmup) {
      w_raw <- NULL
    } else {
      w_raw <- 1 / pmax(g, eps)
      if (stabilize_weights) w_raw <- .stab_w(w_raw)
    }
    
    # fit weighted and unweighted and decide
    f_w <- .fit_f(w_raw, seed_shift = 100*t)
    f_u <- .fit_f(NULL,  seed_shift = 200*t)
    
    # CV-based decision (keep, but allow shutting off by k_cv=0)
    if (k_cv > 1) {
      if (method == "torch") {
        rmse_w <- .cv_rmse(method, X, y, w = w_raw, k = k_cv, seed = seed + 10*t,
                           torch_spec = torch_spec, stabilize_weights = stabilize_weights,
                           alpha = alpha, w_clip_q = w_clip_q, w_min = w_min, w_max = w_max, ...)
        rmse_u <- .cv_rmse(method, X, y, w = NULL,  k = k_cv, seed = seed + 10*t,
                           torch_spec = torch_spec, stabilize_weights = FALSE, ...)
      } else {
        rmse_w <- .cv_rmse(method, X, y, w = w_raw, k = k_cv, seed = seed + 10*t,
                           torch_spec = torch_spec, stabilize_weights = stabilize_weights,
                           alpha = alpha, w_clip_q = w_clip_q, w_min = w_min, w_max = w_max, ...)
        rmse_u <- .cv_rmse(method, X, y, w = NULL,  k = k_cv, seed = seed + 10*t,
                           torch_spec = torch_spec, stabilize_weights = FALSE, ...)
      }
    } else {
      # fast mode: no CV, compare training RMSE (less reliable but fast)
      rmse_w <- .rmse(y, f_w$pred)
      rmse_u <- .rmse(y, f_u$pred)
    }
    
    thresh <- .margin_thresh(rmse_u)
    
    # (5) margin rule + best RMSE rule
    if (!is.null(w_raw) && (rmse_w < (rmse_u - thresh))) {
      f_fit <- f_w$fit
      f_pred <- f_w$pred
      used_weights <- TRUE
    } else {
      f_fit <- f_u$fit
      f_pred <- f_u$pred
      used_weights <- FALSE
    }
    
    # residuals
    r  <- y - f_pred
    r2 <- r^2
    
    # g-step: (7) cross-fitting by default, but can be turned off for speed
    if (crossfit_g) {
      g_pred <- .crossfit_g_pred(r2, seed_shift = 300*t)
      
      # also fit a full-data g_fit object for prediction on NEW data
      if (method == "rfsrc") {
        g_fit <- .train_rfsrc(X, r2, w = NULL, ...)
      } else if (method == "bart") {
        g_fit <- .train_bart(X, r2, w = NULL, seed = seed + 3000 + 300*t, ...)
      } else if (method == "torch") {
        spec <- torch_spec %||% list()
        g_fit <- .train_torch_mlp(X, r2, w = NULL, spec = spec, seed = seed + 4000 + 300*t)
      }
      
    } else {
      # standard (faster) g fit on full data
      if (method == "rfsrc") {
        g_fit <- .train_rfsrc(X, r2, w = NULL, ...)
        g_pred <- .predict_rfsrc(g_fit, X)
      } else if (method == "bart") {
        g_fit <- .train_bart(X, r2, w = NULL, seed = seed + 300*t, ...)
        g_pred <- .predict_bart(g_fit, X)
      } else if (method == "torch") {
        spec <- torch_spec %||% list()
        g_fit <- .train_torch_mlp(X, r2, w = NULL, spec = spec, seed = seed + 300*t)
        g_pred <- .predict_torch_mlp(g_fit, X)
      }
      g_pred <- pmax(as.numeric(g_pred), eps)
    }
    
    g <- pmax(as.numeric(g_pred), eps)
    
    delta <- mean((f_pred - f_pred_prev)^2)
    
    history[[t]] <- list(
      iter = t,
      delta = delta,
      used_weights = used_weights,
      rmse_weighted = rmse_w,
      rmse_unweighted = rmse_u,
      margin_threshold = thresh,
      warmup = warmup,
      crossfit_g = crossfit_g
    )
    
    f_pred_prev <- f_pred
    if (delta < tau) break
  }
  
  # -------------------------
  # D) Variable importance
  # -------------------------
  vimp <- NULL
  if (importance != "none") {
    if (importance == "auto") {
      if (method == "rfsrc") {
        vimp <- .vimp_rfsrc(f_fit)
      } else if (method == "bart") {
        vimp <- .vimp_bart(f_fit, colnames_X = colnames(X))
      } else if (method == "torch") {
        vimp <- .permutation_vimp(.predict_torch_mlp, f_fit, X, y, nrep = vimp_nrep, seed = seed + 999)
      }
    } else if (importance == "permutation") {
      pred_fun <- switch(
        method,
        rfsrc = .predict_rfsrc,
        bart  = .predict_bart,
        torch = .predict_torch_mlp
      )
      vimp <- .permutation_vimp(pred_fun, f_fit, X, y, nrep = vimp_nrep, seed = seed + 999)
    }
  }
  
  out <- list(
    method = method,
    eps = eps,
    tau = tau,
    T = T,
    X_colnames = colnames(X),
    
    baseline = list(
      f_fit = base_f_fit,
      fitted = list(f = as.numeric(base_f_hat))
    ),
    
    varguid = list(
      f_fit = f_fit,
      g_fit = g_fit,
      fitted = list(f = as.numeric(f_pred_prev), g = as.numeric(g)),
      history = do.call(rbind, lapply(history, as.data.frame)),
      vimp = vimp
    )
  )
  class(out) <- "varguid_ml"
  out
}

varguid_predict_ml <- function(object, newdata) {
  stopifnot(inherits(object, "varguid_ml"))
  Xnew <- .make_matrix(newdata)
  
  # Align columns
  if (!is.null(object$X_colnames)) {
    missing <- setdiff(object$X_colnames, colnames(Xnew))
    extra   <- setdiff(colnames(Xnew), object$X_colnames)
    if (length(missing) > 0) {
      for (nm in missing) Xnew <- cbind(Xnew, setNames(rep(0, nrow(Xnew)), nm))
    }
    if (length(extra) > 0) {
      Xnew <- Xnew[, setdiff(colnames(Xnew), extra), drop = FALSE]
    }
    Xnew <- Xnew[, object$X_colnames, drop = FALSE]
  }
  
  # baseline mean
  base_mean <- switch(
    object$method,
    rfsrc = .predict_rfsrc(object$baseline$f_fit, Xnew),
    bart  = .predict_bart(object$baseline$f_fit, Xnew),
    torch = .predict_torch_mlp(object$baseline$f_fit, Xnew)
  )
  
  # varguid mean + variance
  vg_mean <- switch(
    object$method,
    rfsrc = .predict_rfsrc(object$varguid$f_fit, Xnew),
    bart  = .predict_bart(object$varguid$f_fit, Xnew),
    torch = .predict_torch_mlp(object$varguid$f_fit, Xnew)
  )
  vg_var <- switch(
    object$method,
    rfsrc = .predict_rfsrc(object$varguid$g_fit, Xnew),
    bart  = .predict_bart(object$varguid$g_fit, Xnew),
    torch = .predict_torch_mlp(object$varguid$g_fit, Xnew)
  )
  vg_var <- pmax(as.numeric(vg_var), object$eps)
  
  list(
    baseline = list(mean = as.numeric(base_mean)),
    varguid  = list(mean = as.numeric(vg_mean),
                    var  = as.numeric(vg_var),
                    sd   = sqrt(as.numeric(vg_var)))
  )
}


# ============================================================
# VarGuid ML fit (Algorithm 1) — UPDATED with method = "torch"
# ============================================================
# REQUIREMENTS (already defined elsewhere in your script):
#   .make_matrix(), .rmse(), .kfold_ids(), .cv_rmse()
#   .train_rfsrc(), .predict_rfsrc(), .vimp_rfsrc()
#   .train_bart(),  .predict_bart(),  .vimp_bart()
#   .train_dnn(),   .predict_dnn()
#   .train_torch_mlp(), .predict_torch_mlp()
#   .permutation_vimp()
#   `%||%` helper


# install.packages(c("torch", "mlr3", "mlr3learners", "mlr3tuning", "mlr3torch", "paradox"))
# torch::install_torch()
tune_varguid_dnn_torch <- function(x, y, k = 5, seed = 1, n_evals = 30) {
  for (pkg in c("mlr3", "mlr3tuning", "mlr3torch", "paradox")) {
    if (!requireNamespace(pkg, quietly = TRUE)) stop("Need package: ", pkg)
  }
  
  X <- .make_matrix(x)
  dat <- as.data.frame(X)
  dat$y <- as.numeric(y)
  
  task <- mlr3::TaskRegr$new(id = "varguid_dnn_torch", backend = dat, target = "y")
  lrn  <- mlr3::lrn("regr.mlp")
  
  ids <- lrn$param_set$ids()
  
  # ---- Satisfy "required parameters" WITHOUT triggering validation ----
  # Your learner requires these and wants Measure objects; an empty list passes both.
  if ("measures_train" %in% ids) lrn$param_set$values$measures_train <- list()
  if ("measures_valid" %in% ids) lrn$param_set$values$measures_valid <- list()
  
  # Avoid any early-stopping/validation reliance
  if ("patience" %in% ids)  lrn$param_set$values$patience <- 0L
  if ("eval_freq" %in% ids) lrn$param_set$values$eval_freq <- 1  # must be >= 0.5 in your version
  if ("min_delta" %in% ids) lrn$param_set$values$min_delta <- 0
  
  # Fixed defaults
  if ("seed" %in% ids)    lrn$param_set$values$seed <- as.integer(seed)
  if ("device" %in% ids)  lrn$param_set$values$device <- "cpu"
  if ("shuffle" %in% ids) lrn$param_set$values$shuffle <- TRUE
  
  # ---- External search space (random search) ----
  ps <- paradox::ps(
    neurons = paradox::p_int(lower = 16L, upper = 256L),
    n_layers = paradox::p_int(lower = 1L, upper = 4L),
    p = paradox::p_dbl(lower = 0, upper = 0.5),                  # dropout prob
    `opt.lr` = paradox::p_dbl(lower = 1e-4, upper = 5e-3),
    `opt.weight_decay` = paradox::p_dbl(lower = 0, upper = 1e-3),
    epochs = paradox::p_int(lower = 30L, upper = 200L),
    batch_size = paradox::p_int(lower = 32L, upper = 256L)
  )
  
  at <- mlr3tuning::AutoTuner$new(
    learner = lrn,
    resampling = mlr3::rsmp("cv", folds = as.integer(k)),
    measure = mlr3::msr("regr.rmse"),
    search_space = ps,
    terminator = mlr3tuning::trm("evals", n_evals = as.integer(n_evals)),
    tuner = mlr3tuning::tnr("random_search"),
    store_models = FALSE
  )
  
  set.seed(seed)
  at$train(task)
  
  best <- at$tuning_result
  bp <- best$learner_param_vals
  getp <- function(name, default) if (!is.null(bp[[name]])) bp[[name]] else default
  
  neurons   <- as.integer(getp("neurons", 64L))
  n_layers  <- as.integer(getp("n_layers", 2L))
  dropout_p <- as.numeric(getp("p", 0.0))
  lr        <- as.numeric(getp("opt.lr", 1e-3))
  wd        <- as.numeric(getp("opt.weight_decay", 0))
  epochs    <- as.integer(getp("epochs", 100L))
  batch_sz  <- as.integer(getp("batch_size", 64L))
  
  hidden <- rep.int(neurons, n_layers)
  
  list(
    best_spec = list(
      hidden = as.integer(hidden),
      dropout = dropout_p,
      lr = lr,
      epochs = epochs,
      batch_size = batch_sz,
      patience = 0L,
      weight_decay = wd,
      validation_split = 0.2
    ),
    best_rmse = best$perf,
    learner_key = "regr.mlp",
    raw = best
  )
}
