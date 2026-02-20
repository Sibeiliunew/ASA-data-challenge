## ============================================================
## ASA South Florida 2026 Student Data Challenge
## Methods: LASSO, RF, VarGuid-RF, BART, VarGuid-BART
## No variable pre-screening — each method selects its own vars
## 20-split CV + variable selection table + final prediction
## ============================================================

# ---- 0. Load packages ----
library(tidyverse)
library(ggplot2)
library(glmnet)       # LASSO
library(randomForestSRC)  # for importance extraction from rfsrc

source("irlsML1.2.R")

# ---- 1. Download data ----
tmp <- tempfile()
download.file("https://luminwin.github.io/ASASF/train.rds", tmp, mode = "wb")
train_raw <- readRDS(tmp)

download.file("https://luminwin.github.io/ASASF/test.rds", tmp, mode = "wb")
test_raw  <- readRDS(tmp)

# ---- 2. Preprocessing (cleaning only, NO variable selection) ----
outcome_col <- "LBDHDD_outcome"

y_all    <- as.numeric(train_raw[[outcome_col]])
X_all    <- train_raw[, !names(train_raw) %in% outcome_col, drop = FALSE]
X_test_r <- test_raw[, intersect(names(X_all), names(test_raw)), drop = FALSE]

# Combine for consistent encoding
all_X <- rbind(X_all, X_test_r)

# Convert to numeric (factors -> integer codes)
all_X_num <- as.data.frame(lapply(all_X, function(col) {
  if (is.factor(col) || is.character(col)) as.numeric(as.factor(col))
  else as.numeric(col)
}))

# Impute missing with column median
all_X_num <- as.data.frame(lapply(all_X_num, function(col) {
  col[is.na(col)] <- median(col, na.rm = TRUE)
  col
}))

n_tr         <- nrow(X_all)
X_all_clean  <- all_X_num[1:n_tr, ]
X_test_clean <- all_X_num[(n_tr + 1):nrow(all_X_num), ]

X_all_mat    <- as.matrix(X_all_clean)
X_test_mat   <- as.matrix(X_test_clean)
var_names    <- colnames(X_all_mat)

cat("Training:  n =", n_tr, ", p =", ncol(X_all_mat), "\n")
cat("Test:      n =", nrow(X_test_mat), ", p =", ncol(X_test_mat), "\n\n")

# ---- 3. Storage for 50-split CV ----
n_splits <- 20
val_frac  <- 0.25

methods <- c("lasso", "rf_base", "rf_vg", "bart_base", "bart_vg")
rmse_train_mat <- matrix(NA, n_splits, length(methods), dimnames = list(NULL, methods))
rmse_val_mat   <- matrix(NA, n_splits, length(methods), dimnames = list(NULL, methods))

# Variable selection tracking
# For LASSO: binary selected (coef != 0)
# For RF/BART: store importance scores, later threshold to top-K
lasso_selected_counts <- setNames(rep(0L, length(var_names)), var_names)  # how often selected
rf_base_importance    <- setNames(rep(0.0, length(var_names)), var_names)
rf_vg_importance      <- setNames(rep(0.0, length(var_names)), var_names)
bart_base_importance  <- setNames(rep(0.0, length(var_names)), var_names)
bart_vg_importance    <- setNames(rep(0.0, length(var_names)), var_names)

set.seed(2026)
seeds <- sample.int(1e6, n_splits)

cat("--- Starting 20-split cross-validation ---\n")

for (i in seq_len(n_splits)) {
  set.seed(seeds[i])
  val_idx <- sample(seq_len(n_tr), size = floor(val_frac * n_tr))

  Xtr <- X_all_mat[-val_idx, ]
  ytr <- y_all[-val_idx]
  Xte <- X_all_mat[val_idx, ]
  yte <- y_all[val_idx]

  # ---- LASSO ----
  cv_lasso <- tryCatch({
    cv.glmnet(Xtr, ytr, alpha = 1, nfolds = 5)
  }, error = function(e) { message("LASSO error split ", i); NULL })

  if (!is.null(cv_lasso)) {
    # Train RMSE
    pred_lasso_tr <- predict(cv_lasso, newx = Xtr, s = "lambda.min")[, 1]
    rmse_train_mat[i, "lasso"] <- sqrt(mean((ytr - pred_lasso_tr)^2))
    # Val RMSE
    pred_lasso_val <- predict(cv_lasso, newx = Xte, s = "lambda.min")[, 1]
    rmse_val_mat[i, "lasso"] <- sqrt(mean((yte - pred_lasso_val)^2))
    # Track selected variables
    coefs <- coef(cv_lasso, s = "lambda.min")[-1, ]  # drop intercept
    selected <- names(coefs[coefs != 0])
    lasso_selected_counts[selected] <- lasso_selected_counts[selected] + 1L
  }

  # ---- RF ----
  fit_rf <- tryCatch(
    varguid_fit_ml(x = Xtr, y = ytr, method = "rfsrc",
                   T = 10, tau = 1e-6, ntree = 500, nodesize = 5),
    error = function(e) { message("RF error split ", i, ": ", e$message); NULL }
  )
  if (!is.null(fit_rf)) {
    pred_rf_val <- varguid_predict_ml(fit_rf, Xte)
    pred_rf_tr  <- varguid_predict_ml(fit_rf, Xtr)

    rmse_val_mat[i,   "rf_base"] <- sqrt(mean((yte - pred_rf_val$baseline$mean)^2))
    rmse_val_mat[i,   "rf_vg"]   <- sqrt(mean((yte - pred_rf_val$varguid$mean)^2))
    rmse_train_mat[i, "rf_base"] <- sqrt(mean((ytr - pred_rf_tr$baseline$mean)^2))
    rmse_train_mat[i, "rf_vg"]   <- sqrt(mean((ytr - pred_rf_tr$varguid$mean)^2))

    # Extract variable importance from the baseline rfsrc object
    # varguid_fit_ml stores the baseline model; try common slot names
    baseline_model <- tryCatch(fit_rf$baseline, error = function(e) NULL)
    if (!is.null(baseline_model) && inherits(baseline_model, "rfsrc")) {
      imp <- baseline_model$importance
      if (!is.null(imp)) {
        imp_vec <- if (is.matrix(imp)) imp[, 1] else imp
        shared  <- intersect(names(imp_vec), var_names)
        rf_base_importance[shared] <- rf_base_importance[shared] + abs(imp_vec[shared])
      }
    }
    # VarGuid importance: use final-iteration model if available
    vg_model <- tryCatch(fit_rf$varguid_model, error = function(e) NULL)
    if (is.null(vg_model)) vg_model <- tryCatch(fit_rf$final, error = function(e) NULL)
    if (!is.null(vg_model) && inherits(vg_model, "rfsrc")) {
      imp <- vg_model$importance
      if (!is.null(imp)) {
        imp_vec <- if (is.matrix(imp)) imp[, 1] else imp
        shared  <- intersect(names(imp_vec), var_names)
        rf_vg_importance[shared] <- rf_vg_importance[shared] + abs(imp_vec[shared])
      }
    }
  }

  # ---- BART ----
  fit_bart <- tryCatch(
    varguid_fit_ml(x = Xtr, y = ytr, method = "bart",
                   T = 10, tau = 1e-6, ndpost = 800, nskip = 200),
    error = function(e) { message("BART error split ", i, ": ", e$message); NULL }
  )
  if (!is.null(fit_bart)) {
    pred_bart_val <- varguid_predict_ml(fit_bart, Xte)
    pred_bart_tr  <- varguid_predict_ml(fit_bart, Xtr)

    rmse_val_mat[i,   "bart_base"] <- sqrt(mean((yte - pred_bart_val$baseline$mean)^2))
    rmse_val_mat[i,   "bart_vg"]   <- sqrt(mean((yte - pred_bart_val$varguid$mean)^2))
    rmse_train_mat[i, "bart_base"] <- sqrt(mean((ytr - pred_bart_tr$baseline$mean)^2))
    rmse_train_mat[i, "bart_vg"]   <- sqrt(mean((ytr - pred_bart_tr$varguid$mean)^2))

    # BART variable inclusion proportions (avg splits per variable)
    bart_baseline_obj <- tryCatch(fit_bart$baseline, error = function(e) NULL)
    if (!is.null(bart_baseline_obj)) {
      vip <- tryCatch(colMeans(bart_baseline_obj$varcount), error = function(e) NULL)
      if (!is.null(vip)) {
        shared <- intersect(names(vip), var_names)
        if (length(shared) == 0) shared <- var_names[seq_along(vip)]  # fallback by position
        bart_base_importance[shared] <- bart_base_importance[shared] + vip[shared]
      }
    }
    bart_vg_obj <- tryCatch(fit_bart$varguid_model, error = function(e) NULL)
    if (is.null(bart_vg_obj)) bart_vg_obj <- tryCatch(fit_bart$final, error = function(e) NULL)
    if (!is.null(bart_vg_obj)) {
      vip <- tryCatch(colMeans(bart_vg_obj$varcount), error = function(e) NULL)
      if (!is.null(vip)) {
        shared <- intersect(names(vip), var_names)
        if (length(shared) == 0) shared <- var_names[seq_along(vip)]
        bart_vg_importance[shared] <- bart_vg_importance[shared] + vip[shared]
      }
    }
  }

  if (i %% 5 == 0) cat("  Completed split", i, "/", n_splits, "\n")
}

cat("--- Cross-validation complete ---\n\n")

# ---- 4. RMSE Summary Table ----
rmse_train_df <- as.data.frame(rmse_train_mat)
rmse_val_df   <- as.data.frame(rmse_val_mat)

summary_table <- data.frame(
  Method   = methods,
  Train_Mean = round(colMeans(rmse_train_df, na.rm = TRUE), 4),
  Train_SD   = round(apply(rmse_train_df, 2, sd, na.rm = TRUE), 4),
  Val_Mean   = round(colMeans(rmse_val_df,   na.rm = TRUE), 4),
  Val_SD     = round(apply(rmse_val_df,   2, sd, na.rm = TRUE), 4)
)

cat("========== RMSE Summary Table ==========\n")
print(summary_table, row.names = FALSE)
#write.csv(summary_table, "rmse_summary_table.csv", row.names = FALSE)

# ---- 5. Variable Selection Table ----
# LASSO: selection frequency (out of 50 splits)
# RF/BART: average importance score across splits, then rank
# We flag "selected" as: LASSO >= 25/50 splits; others: top 20 by avg importance

top_k <- 10

lasso_freq     <- lasso_selected_counts / n_splits
rf_base_avg    <- rf_base_importance   / n_splits
rf_vg_avg      <- rf_vg_importance     / n_splits
bart_base_avg  <- bart_base_importance / n_splits
bart_vg_avg    <- bart_vg_importance   / n_splits

lasso_selected    <- names(sort(lasso_freq,    decreasing = TRUE)[1:top_k])
rf_base_selected  <- names(sort(rf_base_avg,   decreasing = TRUE)[1:top_k])
rf_vg_selected    <- names(sort(rf_vg_avg,     decreasing = TRUE)[1:top_k])
bart_base_selected<- names(sort(bart_base_avg, decreasing = TRUE)[1:top_k])
bart_vg_selected  <- names(sort(bart_vg_avg,   decreasing = TRUE)[1:top_k])

all_vars_union <- unique(c(lasso_selected, rf_base_selected, rf_vg_selected,
                            bart_base_selected, bart_vg_selected))

var_selection_table <- data.frame(
  Variable      = all_vars_union,
  LASSO_freq    = round(lasso_freq[all_vars_union], 3),
  RF_base_imp   = round(rf_base_avg[all_vars_union], 4),
  RF_VG_imp     = round(rf_vg_avg[all_vars_union], 4),
  BART_base_imp = round(bart_base_avg[all_vars_union], 4),
  BART_VG_imp   = round(bart_vg_avg[all_vars_union], 4),
  LASSO_sel     = all_vars_union %in% lasso_selected,
  RF_base_sel   = all_vars_union %in% rf_base_selected,
  RF_VG_sel     = all_vars_union %in% rf_vg_selected,
  BART_base_sel = all_vars_union %in% bart_base_selected,
  BART_VG_sel   = all_vars_union %in% bart_vg_selected
) %>%
  mutate(
    n_methods_selected = LASSO_sel + RF_base_sel + RF_VG_sel + BART_base_sel + BART_VG_sel
  ) %>%
  arrange(desc(n_methods_selected), desc(LASSO_freq))

cat("\n========== Variable Selection Table (top vars per method) ==========\n")
print(var_selection_table[, c("Variable", "LASSO_freq", "LASSO_sel",
                               "RF_base_sel", "RF_VG_sel",
                               "BART_base_sel", "BART_VG_sel",
                               "n_methods_selected")], row.names = FALSE)

#write.csv(var_selection_table, "variable_selection_table.csv", row.names = FALSE)

# ---- 6. Plots ----

# Plot 1: Validation RMSE Boxplot
method_labels <- c(
  lasso     = "LASSO",
  rf_base   = "RF (Baseline)",
  rf_vg     = "RF (VarGuid)",
  bart_base = "BART (Baseline)",
  bart_vg   = "BART (VarGuid)"
)

p1 <- rmse_val_df %>%
  pivot_longer(everything(), names_to = "Method", values_to = "RMSE") %>%
  mutate(Method = recode(Method, !!!method_labels)) %>%
  ggplot(aes(x = reorder(Method, RMSE, FUN = median), y = RMSE, fill = Method)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21) +
  geom_jitter(width = 0.1, alpha = 0.3, size = 1) +
  coord_flip() +
  labs(title = "Validation RMSE Across 50 Random Splits",
       subtitle = "Lower is better | Each point = one split",
       x = NULL, y = "RMSE") +
  theme_bw(base_size = 13) +
  theme(legend.position = "none")
ggsave("plot1_validation_rmse_boxplot.png", p1, width = 8, height = 5, dpi = 150)

# Plot 2: Train vs Validation RMSE bar chart
p2_data <- summary_table %>%
  pivot_longer(cols = c(Train_Mean, Val_Mean), names_to = "Set", values_to = "Mean_RMSE") %>%
  mutate(
    SD     = ifelse(Set == "Train_Mean", Train_SD, Val_SD),
    Set    = recode(Set, Train_Mean = "Train", Val_Mean = "Validation"),
    Method = recode(Method, !!!method_labels)
  )

p2 <- ggplot(p2_data, aes(x = Method, y = Mean_RMSE, fill = Set)) +
  geom_col(position = position_dodge(0.7), width = 0.6, alpha = 0.85) +
  geom_errorbar(aes(ymin = Mean_RMSE - SD, ymax = Mean_RMSE + SD),
                position = position_dodge(0.7), width = 0.25) +
  labs(title = "Train vs Validation RMSE by Method (Mean ± SD)",
       subtitle = "50 random splits", x = NULL, y = "RMSE", fill = "Dataset") +
  theme_bw(base_size = 13) +
  theme(axis.text.x = element_text(angle = 15, hjust = 1))
ggsave("plot2_train_val_rmse_comparison.png", p2, width = 9, height = 5, dpi = 150)

# Plot 3: Variable selection heatmap (top vars by consensus)
top_vars_plot <- var_selection_table %>%
  filter(n_methods_selected >= 2) %>%
  slice_head(n = 30) %>%
  pull(Variable)

heatmap_data <- data.frame(
  Variable = top_vars_plot,
  LASSO         = lasso_freq[top_vars_plot],
  RF_Baseline   = rf_base_avg[top_vars_plot] / max(rf_base_avg[top_vars_plot] + 1e-9),
  RF_VarGuid    = rf_vg_avg[top_vars_plot]   / max(rf_vg_avg[top_vars_plot] + 1e-9),
  BART_Baseline = bart_base_avg[top_vars_plot] / max(bart_base_avg[top_vars_plot] + 1e-9),
  BART_VarGuid  = bart_vg_avg[top_vars_plot]   / max(bart_vg_avg[top_vars_plot] + 1e-9)
) %>%
  pivot_longer(-Variable, names_to = "Method", values_to = "Score")

p3 <- ggplot(heatmap_data, aes(x = Method, y = reorder(Variable, Score), fill = Score)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "#1d6fa4",
                      name = "Normalized\nImportance") +
  labs(title = "Variable Selection Heatmap",
       subtitle = "Variables selected by ≥2 methods | Scores normalized within method",
       x = NULL, y = NULL) +
  theme_bw(base_size = 11) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))
ggsave("plot3_variable_selection_heatmap.png", p3, width = 8, height = 9, dpi = 150)

# Plot 4: LASSO selection frequency bar chart
p4_data <- data.frame(
  Variable = names(lasso_freq),
  Freq     = as.numeric(lasso_freq)
) %>% filter(Freq > 0) %>% arrange(desc(Freq)) %>% slice_head(n = 30)

p4 <- ggplot(p4_data, aes(x = reorder(Variable, Freq), y = Freq)) +
  geom_col(fill = "#e07b39", alpha = 0.85) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "red") +
  coord_flip() +
  labs(title = "LASSO: Variable Selection Frequency (top 30)",
       subtitle = "Red dashed line = selected in ≥50% of splits",
       x = NULL, y = "Selection Frequency") +
  theme_bw(base_size = 11)
ggsave("plot4_lasso_selection_frequency.png", p4, width = 8, height = 7, dpi = 150)

cat("\nPlots saved.\n\n")

# ---- 7. Select best method & refit on full training data ----
best_idx    <- which.min(summary_table$Val_Mean)
best_method <- summary_table$Method[best_idx]
cat("========== Best method:", best_method,
    "| Val RMSE:", round(summary_table$Val_Mean[best_idx], 4), "==========\n\n")

cat("Refitting on full training data...\n")

if (best_method == "lasso") {
  cv_final   <- cv.glmnet(X_all_mat, y_all, alpha = 1, nfolds = 5)
  final_preds <- predict(cv_final, newx = X_test_mat, s = "lambda.min")[, 1]

} else if (best_method %in% c("rf_base", "rf_vg")) {
  fit_final  <- varguid_fit_ml(x = X_all_mat, y = y_all, method = "rfsrc",
                                T = 10, tau = 1e-6, ntree = 500, nodesize = 5)
  pred_final <- varguid_predict_ml(fit_final, X_test_mat)
  final_preds <- if (best_method == "rf_vg") pred_final$varguid$mean else pred_final$baseline$mean

} else {
  fit_final  <- varguid_fit_ml(x = X_all_mat, y = y_all, method = "bart",
                                T = 10, tau = 1e-6, ndpost = 800, nskip = 200)
  pred_final <- varguid_predict_ml(fit_final, X_test_mat)
  final_preds <- if (best_method == "bart_vg") pred_final$varguid$mean else pred_final$baseline$mean
}

cat("Prediction summary:\n")
print(summary(final_preds))

# ---- 8. Save pred.csv ----
pred_df <- data.frame(pred = as.numeric(final_preds))
stopifnot(nrow(pred_df) == nrow(X_test_mat))
write.csv(pred_df, "pred.csv", row.names = FALSE)

cat("\npred.csv written:", nrow(pred_df), "rows, column name:", names(pred_df), "\n")

# ---- 9. Summary ----
cat("\n========== ALL DONE ==========\n")
cat("Output files:\n")
cat("  pred.csv                          <- competition submission\n")
cat("  rmse_summary_table.csv            <- RMSE comparison across methods\n")
cat("  variable_selection_table.csv      <- selected variables per method\n")
cat("  plot1_validation_rmse_boxplot.png\n")
cat("  plot2_train_val_rmse_comparison.png\n")
cat("  plot3_variable_selection_heatmap.png\n")
cat("  plot4_lasso_selection_frequency.png\n")
cat("Best method:", best_method, "\n")
