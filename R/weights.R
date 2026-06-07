# Phase 1: Adversarial weight learning (Algorithm 1)
# Model-agnostic: operates via backend interface.
#
# Canonical (non-alternating) form: the pooled reference predictor is fit ONCE
# (backend$init_loss gives its per-observation losses); these FIXED losses are
# used for every step. Phase 1 only moves the weights -- the predictor is never
# re-solved inside the loop. The learner is re-estimated only afterwards by
# Phase 2 (.cv_gamma_fit) for the discovered weights.

.learn_weights <- function(backend, gamma_learn, control) {
  eps <- control$eps

  # Step 1: FIXED reference losses from the pooled fit, computed ONCE (Eq. 10).
  # These are the squared residuals of the ridge-OLS pooled predictor and do
  # NOT change across Phase-1 steps.
  init_loss <- backend$init_loss()
  loss_vec <- init_loss

  # Step 2: Initialize weights from the (signed) reference losses (Eq. 10).
  w <- .init_weights(init_loss)
  momentum_buf <- rep(0, length(w))

  # Pre-allocate history
  n_steps <- control$n_steps
  hist <- list(
    risk1 = numeric(n_steps), risk2 = numeric(n_steps),
    objective = numeric(n_steps),
    weight_mean = numeric(n_steps), weight_std = numeric(n_steps),
    weight_entropy = numeric(n_steps)
  )

  for (step in seq_len(n_steps)) {
    # Step 3: Weighted risks and gradients on the FIXED reference losses
    # (Eq. 15-17). No per-step model re-fit.
    terms <- .weighted_risks_and_grads(loss_vec, w, eps)

    # Step 4: Sign of the contrast (Eq. 16)
    sign_diff <- ifelse(terms$risk1 >= terms$risk2, 1.0, -1.0)

    # Step 5: Balance penalty (Eq. 19)
    pen <- .balance_penalty_and_grad(w, control$rho)

    # Step 6: Entropy (Eq. 20)
    ent <- .entropy_and_grad(w, eps)

    # Pure risk contrast |R1 - R2| -- the objective Phase 1 ascends, minus the
    # balance and entropy regularizers.
    objective <- abs(terms$risk1 - terms$risk2) -
                 control$kappa * pen$penalty -
                 control$entropy_coeff * ent$entropy

    # Step 7: Pure-contrast ascent gradient.
    #   d|R1 - R2|/dw = s * (dR1/dw - dR2/dw),  s = sign(R1 - R2).
    # .weighted_risks_and_grads returns grad1 = dR1/dw and
    # grad2 = -dR2_raw/d(1-w) (chain-ruled). The canonical contrast_grad in
    # experiments/acr_core.R computes s*(r1$grad + r2$grad) where r2$grad is the
    # raw d/d(1-w); since our grad2 negates that raw term, the equivalent here is
    # s*(grad1 - grad2).
    contrast_grad <- sign_diff * (terms$grad1 - terms$grad2)
    total_grad <- contrast_grad - control$kappa * pen$grad -
                  control$entropy_coeff * ent$grad

    # Step 8: Momentum update and projection (Eq. 22)
    momentum_buf <- control$momentum * momentum_buf +
                    (1 - control$momentum) * total_grad
    w <- pmin(pmax(w + control$lr * momentum_buf, eps), 1 - eps)

    # Record diagnostics
    hist$risk1[step] <- terms$risk1
    hist$risk2[step] <- terms$risk2
    hist$objective[step] <- objective
    hist$weight_mean[step] <- mean(w)
    hist$weight_std[step] <- sd(w)
    hist$weight_entropy[step] <- ent$entropy

    if (control$verbose && step %% 100 == 0) {
      cat(sprintf("  Step %d/%d: R1=%.4f R2=%.4f |R1-R2|=%.4f w_mean=%.3f\n",
                  step, n_steps, terms$risk1, terms$risk2,
                  abs(terms$risk1 - terms$risk2), mean(w)))
    }
  }

  list(weights = w, history = hist)
}
