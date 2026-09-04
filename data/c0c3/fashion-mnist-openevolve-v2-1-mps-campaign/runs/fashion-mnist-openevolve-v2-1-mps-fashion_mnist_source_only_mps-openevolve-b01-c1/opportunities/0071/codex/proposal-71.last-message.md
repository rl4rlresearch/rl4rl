MECHANISM: Short-horizon tail checkpoint averaging

HYPOTHESIS: Uniformly averaging the final 24 optimizer iterates will exceed 9,247 correct predictions by reducing endpoint minibatch noise without the trajectory lag that caused the 0.99-decay EMA to regress to 9,225.

INTENDED_EDIT: Accumulate each learned parameter over the final 24 post-update checkpoints and replace the endpoint parameters with their uniform average after the last optimizer step.

EVIDENCE: Full-trajectory EMA reduced correctness from 9,247 to 9,225, indicating that broad temporal smoothing is harmful; a tightly localized tail average isolates variance reduction from stale-parameter bias while preserving the proven architecture, supervision, and schedule.

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier
=======
    tail_steps = min(24, total_steps)
    if step >= total_steps - tail_steps:
        with torch.no_grad():
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    state = optimizer.state[parameter]
                    if "_tail_sum" not in state:
                        state["_tail_sum"] = torch.zeros_like(parameter)
                    state["_tail_sum"].add_(parameter)
            if step + 1 >= total_steps:
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        parameter.copy_(
                            optimizer.state[parameter]["_tail_sum"] / tail_steps
                        )
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier
>>>>>>> REPLACE