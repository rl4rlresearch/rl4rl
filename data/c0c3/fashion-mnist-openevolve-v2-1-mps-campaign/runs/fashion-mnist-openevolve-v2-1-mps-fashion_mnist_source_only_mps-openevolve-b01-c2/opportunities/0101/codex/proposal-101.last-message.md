MECHANISM: Late-iterate exponential weight averaging

HYPOTHESIS: A 0.995-decay EMA of the trained parameters will exceed 9,330 correct predictions by reducing variance in the final low-learning-rate iterates without changing the architecture, examples, or model-forward workload.

INTENDED_EDIT: Maintain an EMA inside the optimizer and install the averaged parameters after the final optimizer step; retain the verified training objective and 1.184 evaluation calibration.

EVIDENCE: The current/reference design already verifies the best available score at 9,330 correct with a cosine-decayed learning rate; averaging its late optimization trajectory is a distinct, low-overhead way to improve correctness after evaluation-fusion and loss tweaks yielded no completed contrary metrics.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        betas=(0.9, 0.99),
        weight_decay=5.0e-4,
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        betas=(0.9, 0.99),
        weight_decay=5.0e-4,
    )
    optimizer.ema_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    peak_lr = 3.0e-3
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    torch._foreach_lerp_(optimizer.ema_parameters, parameters, 0.005)
    if step + 1 >= total_steps:
        torch._foreach_copy_(parameters, optimizer.ema_parameters)

    peak_lr = 3.0e-3
>>>>>>> REPLACE