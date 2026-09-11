MECHANISM: Parameter-only sparse late-training EMA

HYPOTHESIS: Averaging learned parameters while retaining final BatchNorm statistics will exceed 9,334 correct predictions by preserving the successful 0.99 EMA’s variance reduction without averaging stateful normalization buffers.

INTENDED_EDIT: Add Reference Design 2’s sparse 0.99 EMA, but apply it only to model parameters using batched foreach operations and leave BatchNorm running statistics at their final trained values.

EVIDENCE: Sparse late-training EMA improved the same architecture from 9,322 to 9,334 correct; isolating parameter averaging tests whether its averaged normalization buffers limit that gain while also reducing snapshot overhead implicated by subsequent EMA timeouts.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=5e-4,
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=5e-4,
    )
    optimizer._averaging_model = model
    optimizer._averaged_parameters = None
    optimizer._averaging_last_step = None
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = PEAK_LR * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = PEAK_LR * multiplier

    should_average = progress >= 0.5 and (
        step % 8 == 0 or step >= total_steps
    )
    if should_average:
        parameters = tuple(optimizer._averaging_model.parameters())
        averaged_parameters = optimizer._averaged_parameters
        last_step = optimizer._averaging_last_step
        with torch.no_grad():
            if averaged_parameters is None:
                averaged_parameters = [
                    parameter.detach().clone()
                    for parameter in parameters
                ]
            else:
                elapsed = max(step - last_step, 1)
                decay = 0.99 ** elapsed
                torch._foreach_mul_(averaged_parameters, decay)
                torch._foreach_add_(
                    averaged_parameters,
                    parameters,
                    alpha=1.0 - decay,
                )
        optimizer._averaged_parameters = averaged_parameters
        optimizer._averaging_last_step = step

        if step >= total_steps:
            with torch.no_grad():
                torch._foreach_copy_(
                    parameters,
                    averaged_parameters,
                )
>>>>>>> REPLACE