# Shared utility functions

.normalize_weights <- function(w, eps = 1e-8) {
  w <- pmax(w, eps)
  w / sum(w)
}

# Weight initialization from per-observation losses (Eq. 10)
# w_i = 0.1 + 0.8 * sigmoid((loss_i - median) / sd)
.init_weights <- function(loss_vec) {
  centered <- loss_vec - median(loss_vec)
  logits <- centered / (sd(centered) + 1e-6)
  0.1 + 0.8 * plogis(logits)
}

# Effective weights for GLM/GAM approximation
# w_eff_i = (1 + gamma_s) * w_tilde1_i + (1 - gamma_s) * w_tilde2_i
.effective_weights <- function(w, gamma_s, eps = 1e-8) {
  w <- pmin(pmax(w, eps), 1 - eps)
  w1 <- w / sum(w)
  w2 <- (1 - w) / sum(1 - w)
  w_eff <- (1 + gamma_s) * w1 + (1 - gamma_s) * w2
  n <- length(w)
  pmax(w_eff * n, eps)  # scale to sum ~ n for glm/gam weights
}

# Post-hoc binarization of soft weights
.harden_weights <- function(w, rho = 0.5) {
  n <- length(w)
  target <- min(max(as.integer(round(rho * n)), 1L), n - 1L)
  hard <- rep(0, n)
  hard[tail(order(w), target)] <- 1
  hard
}

# AUC computation (label-invariant)
.auc_binary <- function(labels, scores) {
  labels <- as.integer(labels)
  n1 <- sum(labels == 1L); n0 <- sum(labels == 0L)
  if (n1 == 0L || n0 == 0L) return(NA_real_)
  r <- rank(scores, ties.method = "average")
  (sum(r[labels == 1L]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

.auc_invariant <- function(labels, scores) {
  a <- .auc_binary(labels, scores)
  if (is.na(a)) NA_real_ else max(a, 1 - a)
}
