MECHANISM: Checkpointed parameter EMA

HYPOTHESIS: An interval-8 approximation of the previously timed-out 0.99-decay EMA will exceed 9,239 correct predictions by retaining its late-trajectory noise suppression with roughly one-eighth the averaging overhead.

INTENDED_EDIT: Maintain parameter EMA checkpoints every eight optimizer steps using decay adjusted for elapsed steps, then install the averaged parameters for validation.

EVIDENCE: Per-step 0.99 EMA timed out before producing accuracy evidence, while subsequent architecture, head, loss, and augmentation changes failed to improve the 9,239-correct cosine-curriculum design; reducing EMA maintenance frequency isolates the still-untested averaging mechanism without changing that design.

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min((step + 1) / max(total_steps, 1), 1.0)
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    current_step = step + 1
    ema_interval = 8
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    with torch.no_grad():
        if current_step == 1:
            for parameter in parameters:
                optimizer.state[parameter]["validation_ema"] = (
                    parameter.detach().clone()
                )
        elif (
            current_step % ema_interval == 0
            or current_step == total_steps
        ):
            previous_ema_step = (
                1
                if current_step <= ema_interval
                else ((current_step - 1) // ema_interval) * ema_interval
            )
            effective_decay = 0.99 ** (
                current_step - previous_ema_step
            )
            ema_parameters = [
                optimizer.state[parameter]["validation_ema"]
                for parameter in parameters
            ]
            torch._foreach_mul_(ema_parameters, effective_decay)
            torch._foreach_add_(
                ema_parameters,
                parameters,
                alpha=1.0 - effective_decay,
            )
        if current_step == total_steps:
            for parameter in parameters:
                parameter.copy_(
                    optimizer.state[parameter]["validation_ema"]
                )

    progress = min(current_step / max(total_steps, 1), 1.0)
>>>>>>> REPLACE