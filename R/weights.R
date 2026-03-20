# Phase 1: Adversarial weight learning (Algorithm 1)
# Model-agnostic: operates via backend interface

.learn_weights <- function(backend, gamma_learn, control) {
  eps <- control$eps

  # Step 1: Initialize weights from initial losses (Eq. 10)
  init_loss <- backend$init_loss()
  w <- .init_weights(init_loss)
  fitted <- backend$fit(w, gamma_learn)
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
    # Step 2: Per-observation losses from current model
    loss_vec <- backend$loss(fitted)

    # Step 3: Weighted risks and gradients (Eq. 15-17)
    terms <- .weighted_risks_and_grads(loss_vec, w, eps)

    # Step 4: Sign determination (Eq. 16)
    sign_diff <- ifelse(terms$risk1 >= terms$risk2, 1.0, -1.0)

    # Step 5: Balance penalty (Eq. 19)
    pen <- .balance_penalty_and_grad(w, control$rho)

    # Step 6: Entropy (Eq. 20)
    ent <- .entropy_and_grad(w, eps)

    # Objective for diagnostics
    objective <- (1 + gamma_learn * sign_diff) * terms$risk1 +
                 (1 - gamma_learn * sign_diff) * terms$risk2 -
                 control$kappa * pen$penalty -
                 control$entropy_coeff * ent$entropy

    # Step 7: Adversary gradient (Eq. 21)
    adv_grad <- (1 + gamma_learn * sign_diff) * terms$grad1 +
                (1 - gamma_learn * sign_diff) * terms$grad2
    total_grad <- adv_grad - control$kappa * pen$grad -
                  control$entropy_coeff * ent$grad

    # Step 8: Momentum update and projection (Eq. 22)
    momentum_buf <- control$momentum * momentum_buf +
                    (1 - control$momentum) * total_grad
    w <- pmin(pmax(w + control$lr * momentum_buf, eps), 1 - eps)

    # Step 9: Re-fit model with updated weights
    fitted <- backend$fit(w, gamma_learn * sign_diff)

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

  list(fitted = fitted, weights = w, history = hist)
}
