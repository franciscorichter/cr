test_that("acr works with linear regression backend", {
  set.seed(42)
  n <- 500
  X1 <- rnorm(n); X2 <- rnorm(n, X1, 0.5)
  Y <- X1 + rnorm(n, 0, 0.5)
  X3 <- Y + c(rep(0, n / 2), rep(2, n / 2)) + rnorm(n, 0, 0.3)
  dat <- data.frame(X1, X2, X3, Y)

  fit <- acr(Y ~ X1 + X2 + X3, data = dat,
             control = acr_control(n_steps = 200, verbose = FALSE))

  expect_s3_class(fit, "acr")
  expect_equal(fit$backend_type, "lm")
  expect_length(fit$weights, n)
  expect_true(all(fit$weights > 0 & fit$weights < 1))
  expect_true(!is.null(fit$gamma_fit))
  expect_true(!is.null(fit$coefficients))
})

test_that("acr works with Poisson GLM backend", {
  set.seed(42)
  n <- 500
  X1 <- rnorm(n)
  Y <- rpois(n, exp(0.5 * X1))
  X2 <- log(Y + 1) + c(rep(0, n / 2), rep(1, n / 2)) + rnorm(n, 0, 0.3)
  dat <- data.frame(X1, X2, Y)

  fit <- acr(Y ~ X1 + X2, data = dat, family = "poisson",
             control = acr_control(n_steps = 100))

  expect_s3_class(fit, "acr")
  expect_equal(fit$backend_type, "glm")
  expect_length(fit$coefficients, 3)  # intercept + 2 vars
})

test_that("acr works with binomial GLM backend", {
  set.seed(42)
  n <- 500
  X1 <- rnorm(n)
  Y <- rbinom(n, 1, plogis(X1))
  X2 <- Y + c(rep(0, n / 2), rep(1, n / 2)) + rnorm(n, 0, 0.3)
  dat <- data.frame(X1, X2, Y)

  fit <- acr(Y ~ X1 + X2, data = dat, family = "binomial",
             control = acr_control(n_steps = 100))

  expect_s3_class(fit, "acr")
  expect_equal(fit$backend_type, "glm")
})

test_that("acr works with GAM backend", {
  set.seed(42)
  n <- 500
  X1 <- rnorm(n)
  Y <- rpois(n, exp(sin(X1)))
  X2 <- log(Y + 1) + c(rep(0, n / 2), rep(1, n / 2)) + rnorm(n, 0, 0.5)
  dat <- data.frame(X1, X2, Y)

  fit <- acr(Y ~ s(X1) + s(X2), data = dat, family = "poisson",
             control = acr_control(n_steps = 50))

  expect_s3_class(fit, "acr")
  expect_equal(fit$backend_type, "gam")
})

test_that("acr_control creates valid control object", {
  ctrl <- acr_control(n_steps = 1000, lr = 0.01, kappa = 0.25)
  expect_equal(ctrl$n_steps, 1000L)
  expect_equal(ctrl$lr, 0.01)
  expect_equal(ctrl$kappa, 0.25)
})

test_that("S3 methods work", {
  set.seed(42)
  n <- 200
  X1 <- rnorm(n); Y <- X1 + rnorm(n, 0, 0.5)
  X2 <- Y + c(rep(0, n / 2), rep(2, n / 2)) + rnorm(n, 0, 0.3)
  dat <- data.frame(X1, X2, Y)

  fit <- acr(Y ~ X1 + X2, data = dat, control = acr_control(n_steps = 50))

  expect_output(print(fit))
  expect_output(summary(fit))
  expect_type(coef(fit), "double")
})
