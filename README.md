# acr

**Adversarial Causal Regularization** without predefined environments.

Learns predictors that are robust to distribution shift by discovering
latent environments and regularizing for causal invariance. Supports
linear regression, GLMs (Poisson, binomial), and GAMs (via mgcv).

## Installation

```r
# install.packages("devtools")
devtools::install_github("franciscorichter/acr")
```

## Quick start

### Linear regression

```r
library(acr)

# Data with environment shift in downstream variable X3
set.seed(42)
n <- 1000
X1 <- rnorm(n); X2 <- rnorm(n, X1, 0.5)
Y <- X1 + rnorm(n, 0, 0.5)
X3 <- Y + c(rep(0, n/2), rep(2, n/2)) + rnorm(n, 0, 0.3)
dat <- data.frame(X1, X2, X3, Y)

fit <- acr(Y ~ X1 + X2 + X3, data = dat)
coef(fit)
summary(fit)
plot(fit)
```

### Poisson GLM

```r
n <- 1000; set.seed(42)
X1 <- rnorm(n)
Y <- rpois(n, exp(0.5 * X1))
X2 <- log(Y + 1) + c(rep(0, n/2), rep(2, n/2)) + rnorm(n, 0, 0.3)
dat <- data.frame(X1, X2, Y)

fit <- acr(Y ~ X1 + X2, data = dat, family = "poisson")
coef(fit)
```

### Binomial GLM

```r
n <- 1000; set.seed(42)
X1 <- rnorm(n)
Y <- rbinom(n, 1, plogis(X1))
X2 <- Y + c(rep(0, n/2), rep(1, n/2)) + rnorm(n, 0, 0.3)
dat <- data.frame(X1, X2, Y)

fit <- acr(Y ~ X1 + X2, data = dat, family = "binomial")
coef(fit)
```

### GAM (nonlinear effects)

```r
n <- 1000; set.seed(42)
X1 <- rnorm(n)
Y <- rpois(n, exp(sin(X1)))
X2 <- log(Y + 1) + c(rep(0, n/2), rep(1, n/2)) + rnorm(n, 0, 0.5)
dat <- data.frame(X1, X2, Y)

fit <- acr(Y ~ s(X1) + s(X2), data = dat, family = "poisson")
coef(fit)
```

## How it works

ACR uses a two-phase algorithm:

**Phase 1: Adversarial weight learning.** Soft weights w_i in [0,1] are
learned for each observation, partitioning the data into two
pseudo-environments. The adversary maximizes the weighted risk difference
(WRD) to discover distribution shifts, while the learner fits a model
that is invariant across the discovered environments.

**Phase 2: Cross-validated gamma selection.** The regularization strength
gamma is selected via K-fold cross-validation using the WRD criterion.
The final model is refitted with the selected gamma.

## Parameters

The main function `acr()` accepts:

- **`formula`** -- Model formula (smooth terms `s()` trigger GAM backend)
- **`data`** -- Data frame
- **`family`** -- `"gaussian"` (default), `"poisson"`, or `"binomial"`
- **`gamma_learn`** -- Adversarial strength for Phase 1 (default 5.0)
- **`gamma_fit`** -- Grid of candidates for Phase 2 CV
- **`control`** -- Fine-grained control via `acr_control()`

Hyperparameters via `acr_control()`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_steps` | 500 | Weight learning iterations |
| `lr` | 0.05 | Learning rate for weight updates |
| `momentum` | 0.9 | Momentum coefficient |
| `kappa` | 0.5 | Balance penalty strength |
| `rho` | 0.5 | Target weight mean |
| `entropy_coeff` | 0.001 | Entropy regularization |
| `ridge` | 0.001 | Ridge penalty |
| `cv_K` | 5 | CV folds |

## Backends

| Family | Backend | Learner step |
|--------|---------|-------------|
| `"gaussian"` | Linear (exact) | Closed-form weighted ridge regression |
| `"poisson"` | GLM | Weighted `glm()` with effective weights |
| `"binomial"` | GLM | Weighted `glm()` with effective weights |
| Smooth terms | GAM | Weighted `mgcv::gam()` with effective weights |

The linear backend gives exact ACR solutions. GLM/GAM backends use an
effective-weight approximation that leverages R's optimized fitting
infrastructure.

## Citation

> Richter, F. and Wit, E.C. (2026). "Adversarial Causal Regularization
> without Predefined Environments." Working paper.

## License

GPL-3
