# Model-agnostic risk computation and gradients (Eqs. 15-21)
# These functions operate on per-observation loss vectors, not on model objects.

# Eq. 15: R_hat_k = sum(w_i * l_i) / sum(w_i)
# Eq. 17: dR_hat/dw_i = (l_i * S - N) / S^2
.normalized_risk_and_grad <- function(loss_vec, weights, eps = 1e-8) {
  w <- pmax(weights, eps)
  total <- sum(w)
  numer <- sum(w * loss_vec)
  risk <- numer / total
  grad <- (loss_vec * total - numer) / (total^2)
  list(risk = risk, grad = grad)
}

# Combined risks and gradients for both pseudo-environments
.weighted_risks_and_grads <- function(loss_vec, w, eps = 1e-8) {
  w <- pmin(pmax(w, eps), 1 - eps)
  w2 <- 1 - w
  r1 <- .normalized_risk_and_grad(loss_vec, w, eps)
  r2_raw <- .normalized_risk_and_grad(loss_vec, w2, eps)
  list(
    risk1 = r1$risk, risk2 = r2_raw$risk,
    grad1 = r1$grad, grad2 = -r2_raw$grad  # chain rule: d(1-w)/dw = -1
  )
}

# Eq. 19: balance penalty Psi(w) = (mean(w) - rho)^2
.balance_penalty_and_grad <- function(w, rho) {
  mean_w <- mean(w)
  penalty <- (mean_w - rho)^2
  grad <- rep(2 * (mean_w - rho) / length(w), length(w))
  list(penalty = penalty, grad = grad)
}

# Eq. 20: entropy H(w) = -mean(w*log(w) + (1-w)*log(1-w))
.entropy_and_grad <- function(w, eps = 1e-8) {
  wc <- pmin(pmax(w, eps), 1 - eps)
  entropy <- -mean(wc * log(wc) + (1 - wc) * log(1 - wc))
  grad <- log(wc) - log(1 - wc)  # dH/dw_i direction (negated in total_grad)
  list(entropy = entropy, grad = grad)
}
