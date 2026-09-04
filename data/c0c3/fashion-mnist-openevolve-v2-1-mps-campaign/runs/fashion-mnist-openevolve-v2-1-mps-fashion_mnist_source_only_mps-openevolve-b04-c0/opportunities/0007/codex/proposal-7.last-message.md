MECHANISM: Second-half exponential weight averaging

HYPOTHESIS: Replacing the final weights with a 0.99-decay EMA accumulated over the second half of training will exceed 9,322 correct predictions by reducing late-optimization variance without changing exposure, architecture, or predictions.

INTENDED_EDIT: Add an optimizer post-step hook that tracks a parameter EMA after halfway and installs it exactly after the final optimizer step.

EVIDENCE: Paired-flip training raised validation_correct from 9,302 to 9,322 and lowered cross-entropy, while batch 64’s extra updates regressed accuracy; this motivates preserving the successful regime and stabilizing its final solution.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=5.0e-4,
        betas=(0.9, 0.99),
        weight_decay=2e-4,
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5.0e-4,
        betas=(0.9, 0.99),
        weight_decay=2e-4,
    )
    ema_start = max(total_steps // 2, 1)
    ema_parameters: list[torch.Tensor] = []
    optimizer_step = 0

    def update_ema(
        _: torch.optim.Optimizer,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        del args, kwargs
        nonlocal ema_parameters, optimizer_step
        optimizer_step += 1
        with torch.no_grad():
            if optimizer_step == ema_start:
                ema_parameters = [
                    parameter.detach().clone()
                    for parameter in model.parameters()
                ]
            elif optimizer_step > ema_start:
                for average, parameter in zip(
                    ema_parameters, model.parameters()
                ):
                    average.lerp_(parameter.detach(), 0.01)

            if optimizer_step == total_steps:
                for parameter, average in zip(
                    model.parameters(), ema_parameters
                ):
                    parameter.copy_(average)

    optimizer.register_step_post_hook(update_ema)
    return optimizer
>>>>>>> REPLACE