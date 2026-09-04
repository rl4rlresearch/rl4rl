MECHANISM: End-of-training exponential weight averaging

HYPOTHESIS: Evaluating a 0.99-decay EMA of learned parameters will exceed 9,331 correct predictions by reducing late-training weight noise while preserving the verified architecture, loss, and inference fusion.

INTENDED_EDIT: Maintain an EMA after every optimizer step and replace model parameters with it after the final step.

EVIDENCE: Adjacent calibration and fusion refinements preserved 9,331 correct but did not improve the score, while asymmetric fusion training reduced correctness to 9,307; parameter averaging tests an orthogonal generalization improvement without changing the successful objective or inference rule.

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = peak_lr * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = peak_lr * multiplier

    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    with torch.no_grad():
        if not hasattr(optimizer, "_evaluation_ema"):
            optimizer._evaluation_ema = [
                parameter.detach().clone() for parameter in parameters
            ]
        else:
            torch._foreach_lerp_(
                optimizer._evaluation_ema,
                parameters,
                0.01,
            )
        if step + 1 >= total_steps:
            torch._foreach_copy_(parameters, optimizer._evaluation_ema)
>>>>>>> REPLACE