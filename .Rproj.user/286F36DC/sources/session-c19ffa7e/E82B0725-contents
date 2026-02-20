### Download Training Data

tmp <- tempfile()
download.file("https://luminwin.github.io/ASASF/train.rds", tmp, mode = "wb")
train <- readRDS(tmp)

### View Variable Labels

lapply(train, attr, "label")

### Download Test Data

download.file("https://luminwin.github.io/ASASF/test.rds", tmp, mode = "wb")
test <- readRDS(tmp)

library(tidyverse)
source("irlsML1.2.R")

Xtr <- train[,2:ncol(train)]
ytr <- train[,1]
Xte <- X[-idx, , drop = FALSE]
yte <- y[-idx]


source("irlsML1.2.R")
# ----- rfsrc -----
fit_rf <- varguid_fit_ml(
  x = Xtr, y = ytr,
  method = "rfsrc",
  T = 10, tau = 1e-6,
  ntree = 500, nodesize = 5
)

pred_rf <- varguid_predict_ml(fit_rf, Xte)

### original Random Forest, no varguid
rmse_rf_base <- sqrt(mean((yte - pred_rf$baseline$mean)^2))
## varguid Random Forest
rmse_rf_vg   <- sqrt(mean((yte - pred_rf$varguid$mean)^2))
c(rmse_baseline = rmse_rf_base, rmse_varguid = rmse_rf_vg)

# ----- bart -----
fit_bart <- varguid_fit_ml(
  x = Xtr, y = ytr,
  method = "bart",
  T = 10, tau = 1e-6,
  ndpost = 800, nskip = 200
)

pred_bart <- varguid_predict_ml(fit_bart, Xte)
rmse_bart_base <- sqrt(mean((yte - pred_bart$baseline$mean)^2))
rmse_bart_vg   <- sqrt(mean((yte - pred_bart$varguid$mean)^2))
c(rmse_baseline = rmse_bart_base, rmse_varguid = rmse_bart_vg)

# ----- Deep learning: torch  -----
## tune_varguid_dnn_torch() only needs to run once for replications
tuned_torch <- tune_varguid_dnn_torch(Xtr, ytr, k = 5, seed = 1, n_evals = 30)

fit_dnn <- varguid_fit_ml(
  x = Xtr, y = ytr,
  method = "torch",
  dnn_spec = tuned_torch$best_spec,
  T = 10, tau = 1e-6
)

### without tuning is not good
#fit_dnn <- varguid_fit_ml(
#  x = Xtr, y = ytr,
#  method = "torch",
#  torch_spec = NULL, 
#  T = 10, tau = 1e-6
#)

pred_torch <- varguid_predict_ml(fit_dnn, Xte)
rmse_torch_base <- sqrt(mean((yte - pred_torch$baseline$mean)^2))
rmse_torch_vg   <- sqrt(mean((yte - pred_torch$varguid$mean)^2))
c(rmse_baseline = rmse_torch_base, rmse_varguid = rmse_torch_vg)



