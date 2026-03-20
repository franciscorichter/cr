#' Adversarial Causal Regularization
#'
#' Discovers causal predictors that are robust to distribution shift using
#' adversarial weight learning, without requiring predefined environment labels.
#'
#' @param formula A formula object. Standard R formula for linear/GLM.
#'   Smooth terms \code{s()} trigger the GAM backend automatically.
#' @param data A data frame.
#' @param family Character: \code{"gaussian"} (default, linear regression),
#'   \code{"poisson"}, or \code{"binomial"}.
#' @param gamma_learn Numeric. Adversarial strength for Phase 1 weight learning.
#'   Higher values produce stronger environment discovery. Default 5.0.
#' @param gamma_fit Numeric vector. Grid of regularization strengths for Phase 2
#'   cross-validation. Default \code{c(0, 0.25, 0.5, 1, 2, 5, 10, 20)}.
#' @param control A list from \code{\link{acr_control}} with algorithm
#'   hyperparameters.
#' @param ... Further arguments passed to the model fitting backend
#'   (\code{glm}, \code{mgcv::gam}).
#' @return An object of class \code{"acr"} with components:
#'   \describe{
#'     \item{coefficients}{Estimated coefficients from the final model.}
#'     \item{weights}{Learned soft environment weights (length n).}
#'     \item{gamma_fit}{Selected regularization strength from Phase 2 CV.}
#'     \item{cv_scores}{Cross-validation scores for each gamma_fit candidate.}
#'     \item{history}{Per-step diagnostics from Phase 1 (risk1, risk2, etc.).}
#'     \item{fitted_model}{The final fitted model object.}
#'     \item{call}{The matched call.}
#'     \item{backend_type}{The backend used: "lm", "glm", or "gam".}
#'   }
#' @references
#' Richter, F. and Wit, E.C. (2026). "Adversarial Causal Regularization
#' without Predefined Environments." Working paper.
#' @importFrom stats glm coef predict residuals as.formula terms.formula
#'   model.matrix model.response model.frame plogis gaussian poisson binomial
#'   median sd setNames
#' @importFrom graphics par lines legend hist
#' @importFrom utils tail
#' @importFrom mgcv gam
#' @examples
#' # Linear regression
#' set.seed(42)
#' n <- 500; X1 <- rnorm(n); X2 <- rnorm(n, X1, 0.5)
#' Y <- X1 + rnorm(n, 0, 0.5)
#' X3 <- Y + c(rep(0, n/2), rep(2, n/2)) + rnorm(n, 0, 0.3)
#' dat <- data.frame(X1, X2, X3, Y)
#' fit <- acr(Y ~ X1 + X2 + X3, data = dat)
#' coef(fit)
#'
#' # Poisson GLM
#' \donttest{
#' set.seed(42)
#' n <- 1000; X1 <- rnorm(n)
#' Y <- rpois(n, exp(X1))
#' X2 <- log(Y + 1) + c(rep(0, n/2), rep(2, n/2)) + rnorm(n, 0, 0.3)
#' dat <- data.frame(X1, X2, Y)
#' fit <- acr(Y ~ X1 + X2, data = dat, family = "poisson")
#' coef(fit)
#' }
#' @export
acr <- function(formula, data, family = c("gaussian", "poisson", "binomial"),
                gamma_learn = 5.0, gamma_fit = NULL,
                control = acr_control(), ...) {
  cl <- match.call()
  family <- match.arg(family)

  if (is.null(gamma_fit))
    gamma_fit <- c(0, 0.25, 0.5, 1, 2, 5, 10, 20)

  gamma_cv <- control$gamma_cv
  if (is.null(gamma_cv)) gamma_cv <- gamma_learn

  # Auto-detect backend from formula and family
  formula_str <- deparse1(formula)
  has_smooth <- grepl("s\\(", formula_str)

  if (family == "gaussian" && !has_smooth) {
    # Linear backend (exact closed-form)
    mf <- model.frame(formula, data)
    y <- model.response(mf)
    X <- model.matrix(formula, data)
    backend <- .make_lm_backend(X, y, control$ridge, control$eps)
    backend_type <- "lm"
  } else if (has_smooth) {
    # GAM backend
    family_obj <- switch(family,
                         gaussian = gaussian(),
                         poisson = poisson(),
                         binomial = binomial())
    backend <- .make_gam_backend(formula, data, family_obj, control$ridge, control$eps)
    backend_type <- "gam"
  } else {
    # GLM backend
    family_obj <- switch(family,
                         poisson = poisson(),
                         binomial = binomial())
    backend <- .make_glm_backend(formula, data, family_obj, control$ridge, control$eps)
    backend_type <- "glm"
  }

  # Phase 1: Adversarial weight learning
  phase1 <- .learn_weights(backend, gamma_learn, control)

  # Phase 2: CV for gamma_fit selection
  phase2 <- .cv_gamma_fit(backend, phase1$weights, gamma_fit, gamma_cv, control)

  structure(
    list(
      coefficients = backend$coef_fn(phase2$fitted),
      weights = phase1$weights,
      gamma_fit = phase2$best_gamma,
      gamma_learn = gamma_learn,
      cv_scores = setNames(phase2$cv_scores, gamma_fit),
      history = phase1$history,
      fitted_model = phase2$fitted,
      call = cl,
      backend_type = backend_type,
      control = control
    ),
    class = "acr"
  )
}
