#' @export
print.acr <- function(x, ...) {
  cat("Adversarial Causal Regularization\n\n")
  cat("Call:\n  ", deparse(x$call), "\n\n")
  cat("Backend:", x$backend_type, "\n")
  cat("Gamma (learn):", x$gamma_learn, "\n")
  cat("Gamma (fit, CV-selected):", x$gamma_fit, "\n\n")
  cat("Coefficients:\n")
  print(round(x$coefficients, 4))
  cat("\nWeight summary: mean =", round(mean(x$weights), 3),
      ", sd =", round(sd(x$weights), 3), "\n")
  invisible(x)
}

#' @export
summary.acr <- function(object, ...) {
  cat("Adversarial Causal Regularization\n\n")
  cat("Call:\n  ", deparse(object$call), "\n\n")
  cat("Backend:", object$backend_type, "\n")
  cat("Phase 1: gamma_learn =", object$gamma_learn,
      ",", object$control$n_steps, "steps\n")
  cat("Phase 2: gamma_fit =", object$gamma_fit,
      "(selected from", length(object$cv_scores), "candidates)\n\n")

  cat("Coefficients:\n")
  beta <- object$coefficients
  rel <- abs(beta) / max(abs(beta) + 1e-10)
  df <- data.frame(Estimate = round(beta, 4), RelMagnitude = round(rel, 3))
  print(df)

  cat("\nCV scores (lower = better):\n")
  print(round(object$cv_scores, 4))

  cat("\nWeight distribution:\n")
  cat("  Mean:", round(mean(object$weights), 3), "\n")
  cat("  SD:", round(sd(object$weights), 3), "\n")
  h <- object$history
  cat("  Final risk gap |R1-R2|:",
      round(abs(tail(h$risk1, 1) - tail(h$risk2, 1)), 4), "\n")

  invisible(object)
}

#' @export
coef.acr <- function(object, ...) object$coefficients

#' @export
predict.acr <- function(object, newdata, ...) {
  if (object$backend_type == "lm") {
    X_new <- model.matrix(object$call$formula, newdata)
    as.numeric(X_new %*% object$coefficients)
  } else {
    predict(object$fitted_model, newdata = newdata, type = "response")
  }
}

#' @export
plot.acr <- function(x, ...) {
  h <- x$history
  n <- length(h$risk1)
  oldpar <- par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  on.exit(par(oldpar))

  plot(seq_len(n), h$risk1, type = "l", col = "blue",
       xlab = "Step", ylab = "Risk", main = "Risks")
  lines(seq_len(n), h$risk2, col = "red")
  legend("topright", c("R1", "R2"), col = c("blue", "red"), lty = 1, cex = 0.7)

  plot(seq_len(n), abs(h$risk1 - h$risk2), type = "l", col = "purple",
       xlab = "Step", ylab = "|R1 - R2|", main = "Risk gap")

  plot(seq_len(n), h$weight_mean, type = "l", col = "darkgreen",
       xlab = "Step", ylab = "Mean weight", main = "Weight mean", ylim = c(0, 1))

  hist(x$weights, breaks = 30, main = "Weight distribution",
       xlab = "Weight", col = "lightblue", border = "gray")
}
