MECHANISM: Late-stage exponential weight averaging

HYPOTHESIS: Averaging the final 30% of optimization with decay 0.995 will exceed 9,319 correct predictions by reducing parameter noise while preserving the verified training trajectory and regularization.

INTENDED_EDIT: Maintain an EMA of learned parameters during the final 30% of training and install the averaged weights after the last optimizer step.

EVIDENCE: Late label-smoothing taper reduced accuracy from 9,319 to 9,307, favoring unchanged loss dynamics; EMA instead regularizes the final solution without altering gradients, exposure, architecture, or batch composition.

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier

    if progress >= 0.70:
        decay = 0.995
        with torch.no_grad():
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    state = optimizer.state[parameter]
                    if "parameter_ema" not in state:
                        state["parameter_ema"] = parameter.detach().clone()
                    else:
                        state["parameter_ema"].mul_(decay).add_(
                            parameter, alpha=1.0 - decay
                        )

            if step >= total_steps:
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        parameter.copy_(
                            optimizer.state[parameter]["parameter_ema"]
                        )
>>>>>>> REPLACE