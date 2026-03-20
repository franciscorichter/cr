# GLM backend (Poisson, binomial) using effective-weight approximation
# Suppress NOTE about .acr_w: it's a column added to data at runtime
utils::globalVariables(".acr_w")

.make_glm_backend <- function(formula, data, family_obj, ridge = 1e-3, eps = 1e-6) {
  list(
    fit = function(w, gamma_s) {
      w_eff <- .effective_weights(w, gamma_s, eps)
      fit_data <- data
      fit_data$.acr_w <- w_eff
      glm(formula, family = family_obj, data = fit_data, weights = .acr_w)
    },
    loss = function(fitted) {
      family_obj$dev.resids(fitted$y, fitted$fitted.values, rep(1, length(fitted$y)))
    },
    predict_fn = function(fitted, newdata) {
      predict(fitted, newdata = newdata, type = "response")
    },
    coef_fn = function(fitted) coef(fitted),
    init_loss = function() {
      fit0 <- glm(formula, family = family_obj, data = data)
      family_obj$dev.resids(fit0$y, fit0$fitted.values, rep(1, length(fit0$y)))
    },
    type = "glm"
  )
}
