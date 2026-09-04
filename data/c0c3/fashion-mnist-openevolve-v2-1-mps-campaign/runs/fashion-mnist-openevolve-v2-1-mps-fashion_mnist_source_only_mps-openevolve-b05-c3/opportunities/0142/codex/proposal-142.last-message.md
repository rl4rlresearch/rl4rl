MECHANISM: Terminal label-smoothing annealing

HYPOTHESIS: Gradually reducing label smoothing from 0.02 to zero during the validation-aligned terminal phase will exceed 9,206 correct predictions by refining learned class boundaries without an abrupt loss transition; verified 1.10× outer calibration will improve tied-count cross-entropy without changing argmaxes.

INTENDED_EDIT: Linearly anneal label smoothing only during the final 13/32 of training and replace the final 1.05× ensemble-logit multiplier with the verified 1.10× value.

EVIDENCE: Inference-only calibration preserved exactly 9,206 correct while clean-view weighting fell to 9,202, indicating that further correctness gains likely require changed learned boundaries. The terminal hard-label experiment supplied no performance result, so gradual annealing tests its mechanism more conservatively; Reference Design 3 verifies 1.10× outer calibration at 9,206 correct and 0.224459 cross-entropy.

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE

<<<<<<< SEARCH
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
) -> torch.Tensor:
    if step * 32 < total_steps * 19:
        smoothing = 0.02
    else:
        terminal_progress = (
            step * 32 - total_steps * 19
        ) / max(total_steps * 13, 1)
        smoothing = 0.02 * max(0.0, 1.0 - terminal_progress)
    return F.cross_entropy(model(images), labels, label_smoothing=smoothing)
>>>>>>> REPLACE