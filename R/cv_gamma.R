# Phase 2: K-fold CV for gamma_fit selection

.cv_gamma_fit <- function(backend, w, fit_gammas, gamma_cv, control) {
  n <- length(w)
  eps <- control$eps
  K <- control$cv_K

  set.seed(control$cv_seed)
  folds <- sample(rep(seq_len(K), length.out = n))

  cv_scores <- matrix(0, nrow = length(fit_gammas), ncol = K)

  for (k in seq_len(K)) {
    te_mask <- folds == k

    for (j in seq_along(fit_gammas)) {
      fg <- fit_gammas[j]

      # Determine sign from current weights on training data
      fitted_j <- backend$fit(w, fg)
      loss_vec <- backend$loss(fitted_j)

      # Compute WRD on held-out fold using held-out weights
      loss_te <- loss_vec[te_mask]
      w_te <- pmin(pmax(w[te_mask], eps), 1 - eps)
      w_te2 <- 1 - w_te

      R1 <- sum(w_te * loss_te) / sum(w_te)
      R2 <- sum(w_te2 * loss_te) / sum(w_te2)
      cv_scores[j, k] <- (R1 + R2) + gamma_cv * abs(R1 - R2)
    }
  }

  mean_scores <- rowMeans(cv_scores)
  best_idx <- which.min(mean_scores)
  best_gamma <- fit_gammas[best_idx]

  # Refit on full data with best gamma
  # Determine sign from fitted residuals
  fitted_best <- backend$fit(w, best_gamma)
  loss_best <- backend$loss(fitted_best)
  rg <- .weighted_risks_and_grads(loss_best, w, eps)
  s <- ifelse(rg$risk1 >= rg$risk2, 1, -1)
  final_fitted <- backend$fit(w, best_gamma * s)

  list(fitted = final_fitted, best_gamma = best_gamma,
       cv_scores = mean_scores, fit_gammas = fit_gammas)
}
