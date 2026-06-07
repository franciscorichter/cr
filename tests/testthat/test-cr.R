test_that("acr works with linear regression backend", {
  set.seed(42)
  n <- 500
  X1 <- rnorm(n); X2 <- rnorm(n, X1, 0.5)
  Y <- X1 + rnorm(n, 0, 0.5)
  X3 <- Y + c(rep(0, n / 2), rep(2, n / 2)) + rnorm(n, 0, 0.3)
  dat <- data.frame(X1, X2, X3, Y)

  fit <- acr(Y ~ X1 + X2 + X3, data = dat,
             control = cr_control(n_steps = 200, verbose = FALSE))

  expect_s3_class(fit, "cr")
  expect_equal(fit$backend_type, "lm")
  expect_equal(fit$mode, "acr")
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
             control = cr_control(n_steps = 100))

  expect_s3_class(fit, "cr")
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
             control = cr_control(n_steps = 100))

  expect_s3_class(fit, "cr")
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
             control = cr_control(n_steps = 50))

  expect_s3_class(fit, "cr")
  expect_equal(fit$backend_type, "gam")
})

test_that("cr() with known environments runs causal regularization", {
  set.seed(42)
  n <- 500
  X1 <- rnorm(n); X2 <- rnorm(n, X1, 0.5)
  Y <- X1 + rnorm(n, 0, 0.5)
  X3 <- Y + c(rep(0, n / 2), rep(2, n / 2)) + rnorm(n, 0, 0.3)
  dat <- data.frame(X1, X2, X3, Y)
  env_factor <- factor(c(rep("A", n / 2), rep("B", n / 2)))

  fit <- cr(Y ~ ., data = dat, env = env_factor,
            control = cr_control(n_steps = 10))

  expect_s3_class(fit, "cr")
  expect_equal(fit$mode, "cr")
  expect_true(is.null(fit$history))
  expect_true(is.na(fit$gamma_learn))
  expect_length(fit$weights, n)
  # intercept + X1 + X2 + X3 = 4 coefficients
  expect_length(fit$coefficients, 4)
})

test_that("cr() rejects environments without exactly two levels", {
  set.seed(42)
  n <- 60
  X1 <- rnorm(n); Y <- X1 + rnorm(n, 0, 0.5)
  dat <- data.frame(X1, Y)
  env3 <- factor(rep(c("A", "B", "C"), length.out = n))
  expect_error(cr(Y ~ X1, data = dat, env = env3),
               "exactly two levels")
})

test_that("cr_control creates valid control object", {
  ctrl <- cr_control(n_steps = 1000, lr = 0.01, kappa = 0.25)
  expect_equal(ctrl$n_steps, 1000L)
  expect_equal(ctrl$lr, 0.01)
  expect_equal(ctrl$kappa, 0.25)
})

test_that("acr_control alias still works", {
  ctrl <- acr_control(n_steps = 50)
  expect_equal(ctrl$n_steps, 50L)
})

test_that("S3 methods work", {
  set.seed(42)
  n <- 200
  X1 <- rnorm(n); Y <- X1 + rnorm(n, 0, 0.5)
  X2 <- Y + c(rep(0, n / 2), rep(2, n / 2)) + rnorm(n, 0, 0.3)
  dat <- data.frame(X1, X2, Y)

  fit <- acr(Y ~ X1 + X2, data = dat, control = cr_control(n_steps = 50))

  expect_output(print(fit))
  expect_output(summary(fit))
  expect_type(coef(fit), "double")
})
