# GAM backend using effective-weight approximation

.make_gam_backend <- function(formula, data, family_obj, ridge = 1e-3, eps = 1e-6) {
  list(
    fit = function(w, gamma_s) {
      w_eff <- .effective_weights(w, gamma_s, eps)
      fit_data <- data
      fit_data$.cr_w <- w_eff
      mgcv::gam(formula, family = family_obj, data = fit_data, weights = .cr_w)
    },
    loss = function(fitted) {
      family_obj$dev.resids(fitted$y, fitted$fitted.values, rep(1, length(fitted$y)))
    },
    predict_fn = function(fitted, newdata) {
      as.numeric(predict(fitted, newdata = newdata, type = "response"))
    },
    coef_fn = function(fitted) coef(fitted),
    init_loss = function() {
      fit0 <- mgcv::gam(formula, family = family_obj, data = data)
      family_obj$dev.resids(fit0$y, fit0$fitted.values, rep(1, length(fit0$y)))
    },
    type = "gam"
  )
}
