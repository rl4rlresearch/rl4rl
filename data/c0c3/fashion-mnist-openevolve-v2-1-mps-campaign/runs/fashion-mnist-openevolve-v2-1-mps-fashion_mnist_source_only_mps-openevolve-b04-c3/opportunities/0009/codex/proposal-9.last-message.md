MECHANISM: Low-overhead exponential weight averaging

HYPOTHESIS: Averaging the final trajectory of the qualified batch-64 model will exceed 9,229 correct predictions by reducing small-batch parameter variance without increasing examples or spatial computation enough to risk timeout.

INTENDED_EDIT: Maintain an exponential moving average of trainable parameters, update it every four optimizer steps, and install the averaged parameters after the final step; retain the qualified architecture, augmentation, loss, batch size, and learning-rate schedule.

EVIDENCE: Batch size 64 improved correctness from 9,210 to 9,229, while attempts to gain more updates at batch sizes 56, 48, and 32 timed out; parameter averaging targets the noisier small-batch trajectory without adding forward or backward passes.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=2.5e-3,
        weight_decay=1e-4,
    )
    optimizer._ema_model_parameters = parameters
    optimizer._ema_parameters = [
        parameter.detach().clone() for parameter in parameters
    ]
    optimizer._ema_step = 0
    optimizer._ema_total_steps = total_steps
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = step / max(total_steps, 1)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = step / max(total_steps, 1)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier

    optimizer._ema_step += 1
    is_final_step = optimizer._ema_step >= optimizer._ema_total_steps
    if optimizer._ema_step % 4 == 0 or is_final_step:
        with torch.no_grad():
            torch._foreach_lerp_(
                optimizer._ema_parameters,
                optimizer._ema_model_parameters,
                0.04,
            )
            if is_final_step:
                torch._foreach_copy_(
                    optimizer._ema_model_parameters,
                    optimizer._ema_parameters,
                )
>>>>>>> REPLACE