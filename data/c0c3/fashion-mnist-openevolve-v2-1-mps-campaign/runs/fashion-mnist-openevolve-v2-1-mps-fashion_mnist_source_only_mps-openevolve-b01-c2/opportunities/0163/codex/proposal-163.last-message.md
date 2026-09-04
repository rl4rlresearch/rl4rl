MECHANISM: Periodic end-of-training exponential weight averaging

HYPOTHESIS: A computationally sparse approximation of the previously unverified 0.99-decay EMA will exceed 9,331 correct predictions without breaching the training-time limit.

INTENDED_EDIT: Restore the best verified flip-fusion coefficients and update an equivalent 0.99-per-step EMA every 16 steps, installing the averaged parameters after the final optimizer step.

EVIDENCE: Fine-grained fusion and calibration changes have saturated at 9,331 correct, while per-step EMA timed out without performance evidence; periodic updates test the same orthogonal generalization mechanism with substantially less overhead.

<<<<<<< SEARCH
            logits = 1.184 * (
                0.49149951171875 * logits + 0.50850048828125 * flipped_logits
            )
=======
            logits = 1.184 * (
                0.4914990234375 * logits + 0.5085009765625 * flipped_logits
            )
>>>>>>> REPLACE

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
    optimizer.ema_interval = 16
    optimizer.ema_decay = 0.99 ** optimizer.ema_interval
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
    finished = step + 1 >= total_steps
    if (step + 1) % optimizer.ema_interval == 0 or finished:
        with torch.no_grad():
            parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            for average, parameter in zip(
                optimizer.ema_parameters,
                parameters,
            ):
                average.lerp_(parameter.detach(), 1.0 - optimizer.ema_decay)
            if finished:
                for parameter, average in zip(
                    parameters,
                    optimizer.ema_parameters,
                ):
                    parameter.copy_(average)

    peak_lr = 3.0e-3
>>>>>>> REPLACE