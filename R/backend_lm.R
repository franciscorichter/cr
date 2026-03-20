# Linear regression backend (exact closed-form solution)

.make_lm_backend <- function(X, y, ridge = 1e-3, eps = 1e-6) {
  n <- nrow(X); p <- ncol(X)

  # Eq. 12-13: weighted moments
  weighted_moments <- function(w) {
    w_norm <- .normalize_weights(w, eps)
    WX <- X * w_norm
    G <- crossprod(X, WX)
    Z <- crossprod(X, w_norm * y)
    list(G = G, Z = Z)
  }

  # Eq. 14: closed-form beta from weights
  solve_beta <- function(w, gamma_s) {
    w <- pmin(pmax(w, eps), 1 - eps)
    m1 <- weighted_moments(w)
    m2 <- weighted_moments(1 - w)
    G_plus <- m1$G + m2$G
    G_delta <- m1$G - m2$G
    Z_plus <- m1$Z + m2$Z
    Z_delta <- m1$Z - m2$Z
    A <- G_plus + gamma_s * G_delta + ridge * diag(p)
    b <- Z_plus + gamma_s * Z_delta
    as.numeric(solve(A, b))
  }

  list(
    fit = function(w, gamma_s) {
      list(coefficients = solve_beta(w, gamma_s))
    },
    loss = function(fitted) {
      as.numeric((y - X %*% fitted$coefficients)^2)
    },
    predict_fn = function(fitted, newdata) {
      if (is.data.frame(newdata)) newdata <- as.matrix(newdata)
      as.numeric(newdata %*% fitted$coefficients)
    },
    coef_fn = function(fitted) fitted$coefficients,
    init_loss = function() {
      beta_ols <- as.numeric(solve(crossprod(X) + ridge * diag(p), crossprod(X, y)))
      as.numeric((y - X %*% beta_ols)^2)
    },
    type = "lm"
  )
}
