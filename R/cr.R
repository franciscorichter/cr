# Internal: auto-detect and build the fitting backend from formula + family.
.build_backend <- function(formula, data, family, control) {
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

  list(backend = backend, backend_type = backend_type)
}

#' Causal Regularization (with or without predefined environments)
#'
#' Learns predictors that are robust to distribution shift via causal
#' regularization. The function operates in two modes that share the same
#' fitting machinery and differ only in how the soft environment weights are
#' obtained.
#'
#' \strong{Causal Regularization (CR), environments known.} When \code{env} is
#' supplied (a two-level factor of environment labels), the soft weights are
#' fixed directly from the labels and the adversarial weight-learning phase is
#' skipped. Only the cross-validated selection of the regularization strength
#' (Phase 2) is performed.
#'
#' \strong{Adversarial Causal Regularization (ACR), environments unknown.} When
#' \code{env} is \code{NULL}, the soft environment weights are discovered
#' adversarially in Phase 1, after which the same cross-validation step
#' (Phase 2) selects the regularization strength.
#'
#' @param formula A formula object. Standard R formula for linear/GLM.
#'   Smooth terms \code{s()} trigger the GAM backend automatically.
#' @param data A data frame.
#' @param env Optional environment labels. If \code{NULL} (default), the
#'   environments are discovered adversarially (ACR mode). If supplied, it is
#'   coerced to a factor and must have exactly two levels; the soft weights are
#'   then fixed from the labels (CR mode) and Phase 1 is skipped.
#' @param family Character: \code{"gaussian"} (default, linear regression),
#'   \code{"poisson"}, or \code{"binomial"}.
#' @param gamma_learn Numeric. Adversarial strength for Phase 1 weight learning
#'   (ACR mode only). Higher values produce stronger environment discovery.
#'   Default 5.0. Ignored (and reported as \code{NA}) in CR mode.
#' @param gamma_fit Numeric vector. Grid of regularization strengths for Phase 2
#'   cross-validation. Default \code{c(0, 0.25, 0.5, 1, 2, 5, 10, 20)}.
#' @param control A list from \code{\link{cr_control}} with algorithm
#'   hyperparameters.
#' @param ... Further arguments passed to the model fitting backend
#'   (\code{glm}, \code{mgcv::gam}).
#' @return An object of class \code{"cr"} with components:
#'   \describe{
#'     \item{coefficients}{Estimated coefficients from the final model.}
#'     \item{weights}{Soft environment weights (length n); learned in ACR mode,
#'       fixed from \code{env} in CR mode.}
#'     \item{gamma_fit}{Selected regularization strength from Phase 2 CV.}
#'     \item{gamma_learn}{Adversarial strength used in Phase 1 (\code{NA} in CR
#'       mode).}
#'     \item{cv_scores}{Cross-validation scores for each gamma_fit candidate.}
#'     \item{history}{Per-step Phase 1 diagnostics (\code{NULL} in CR mode).}
#'     \item{environments}{The environment factor in CR mode, \code{NULL} in ACR
#'       mode.}
#'     \item{mode}{\code{"acr"} or \code{"cr"}.}
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
#' # ACR mode: environments unknown, discovered adversarially
#' set.seed(42)
#' n <- 500; X1 <- rnorm(n); X2 <- rnorm(n, X1, 0.5)
#' Y <- X1 + rnorm(n, 0, 0.5)
#' X3 <- Y + c(rep(0, n/2), rep(2, n/2)) + rnorm(n, 0, 0.3)
#' dat <- data.frame(X1, X2, X3, Y)
#' fit <- cr(Y ~ X1 + X2 + X3, data = dat)
#' coef(fit)
#'
#' # CR mode: environments known, supplied as a two-level factor
#' env <- factor(c(rep("A", n/2), rep("B", n/2)))
#' fit_cr <- cr(Y ~ X1 + X2 + X3, data = dat, env = env)
#' coef(fit_cr)
#'
#' # Poisson GLM (ACR mode)
#' \donttest{
#' set.seed(42)
#' n <- 1000; X1 <- rnorm(n)
#' Y <- rpois(n, exp(X1))
#' X2 <- log(Y + 1) + c(rep(0, n/2), rep(2, n/2)) + rnorm(n, 0, 0.3)
#' dat <- data.frame(X1, X2, Y)
#' fit <- cr(Y ~ X1 + X2, data = dat, family = "poisson")
#' coef(fit)
#' }
#' @export
cr <- function(formula, data, env = NULL,
               family = c("gaussian", "poisson", "binomial"),
               gamma_learn = 5.0, gamma_fit = NULL,
               control = cr_control(), ...) {
  cl <- match.call()
  family <- match.arg(family)

  if (is.null(gamma_fit))
    gamma_fit <- c(0, 0.25, 0.5, 1, 2, 5, 10, 20)

  gamma_cv <- control$gamma_cv
  if (is.null(gamma_cv)) gamma_cv <- gamma_learn

  # Auto-detect and build the backend
  bb <- .build_backend(formula, data, family, control)
  backend <- bb$backend
  backend_type <- bb$backend_type

  if (is.null(env)) {
    # ACR mode: Phase 1 discovers the environment weights adversarially
    phase1 <- .learn_weights(backend, gamma_learn, control)
    w <- phase1$weights
    history <- phase1$history
    mode <- "acr"
    environments <- NULL
  } else {
    # CR mode: weights fixed from the supplied (two-level) environment labels
    env <- as.factor(env)
    if (nlevels(env) != 2)
      stop("cr(): `env` must have exactly two levels (two environments).")
    w <- as.numeric(env == levels(env)[2])
    w <- pmin(pmax(w, control$eps), 1 - control$eps)
    history <- NULL
    mode <- "cr"
    environments <- env
    gamma_learn <- NA
  }

  # Phase 2: CV for gamma_fit selection (identical in both modes)
  phase2 <- .cv_gamma_fit(backend, w, gamma_fit, gamma_cv, control)

  structure(
    list(
      coefficients = backend$coef_fn(phase2$fitted),
      weights = w,
      gamma_fit = phase2$best_gamma,
      gamma_learn = gamma_learn,
      cv_scores = setNames(phase2$cv_scores, gamma_fit),
      history = history,
      environments = environments,
      mode = mode,
      fitted_model = phase2$fitted,
      call = cl,
      backend_type = backend_type,
      control = control
    ),
    class = "cr"
  )
}

#' Adversarial Causal Regularization (alias)
#'
#' \code{acr()} is a thin alias for \code{\link{cr}} with \code{env = NULL},
#' i.e. the adversarial mode in which environment labels are unknown and are
#' discovered by learning soft weights adversarially. All other arguments are
#' passed through to \code{cr()}.
#'
#' @param formula A formula object (see \code{\link{cr}}).
#' @param data A data frame.
#' @param ... Further arguments passed to \code{\link{cr}} (e.g. \code{family},
#'   \code{gamma_learn}, \code{gamma_fit}, \code{control}).
#' @return An object of class \code{"cr"} (see \code{\link{cr}}).
#' @references
#' Richter, F. and Wit, E.C. (2026). "Adversarial Causal Regularization
#' without Predefined Environments." Working paper.
#' @seealso \code{\link{cr}}
#' @examples
#' set.seed(42)
#' n <- 500; X1 <- rnorm(n); X2 <- rnorm(n, X1, 0.5)
#' Y <- X1 + rnorm(n, 0, 0.5)
#' X3 <- Y + c(rep(0, n/2), rep(2, n/2)) + rnorm(n, 0, 0.3)
#' dat <- data.frame(X1, X2, X3, Y)
#' fit <- acr(Y ~ X1 + X2 + X3, data = dat)
#' coef(fit)
#' @export
acr <- function(formula, data, ...) {
  cl <- match.call()
  fit <- cr(formula, data, env = NULL, ...)
  fit$call <- cl
  fit
}
