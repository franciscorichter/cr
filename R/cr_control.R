#' Control parameters for causal regularization
#'
#' @param n_steps Number of adversarial weight update steps (Phase 1, ACR mode).
#' @param lr Learning rate for weight updates.
#' @param momentum Momentum coefficient for weight updates.
#' @param kappa Balance penalty coefficient.
#' @param rho Target weight mean (balance point).
#' @param entropy_coeff Entropy regularization coefficient.
#' @param ridge Ridge penalty for model fitting.
#' @param eps Numerical stability floor.
#' @param cv_K Number of folds for Phase 2 cross-validation.
#' @param cv_seed Seed for CV fold assignment.
#' @param gamma_cv Gamma used in the WRD criterion during CV. Defaults to gamma_learn.
#' @param verbose Print progress during weight learning.
#' @return A list of control parameters.
#' @export
cr_control <- function(n_steps = 500L, lr = 0.05, momentum = 0.9,
                       kappa = 0.5, rho = 0.5, entropy_coeff = 0.001,
                       ridge = 1e-3, eps = 1e-6,
                       cv_K = 5L, cv_seed = 42L,
                       gamma_cv = NULL,
                       verbose = FALSE) {
  list(
    n_steps = as.integer(n_steps), lr = lr, momentum = momentum,
    kappa = kappa, rho = rho, entropy_coeff = entropy_coeff,
    ridge = ridge, eps = eps,
    cv_K = as.integer(cv_K), cv_seed = cv_seed,
    gamma_cv = gamma_cv, verbose = verbose
  )
}

#' @rdname cr_control
#' @export
acr_control <- cr_control
