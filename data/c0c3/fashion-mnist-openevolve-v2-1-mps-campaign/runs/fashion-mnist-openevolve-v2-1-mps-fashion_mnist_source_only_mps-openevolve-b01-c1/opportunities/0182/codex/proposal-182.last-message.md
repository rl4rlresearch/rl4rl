MECHANISM: Late-training exponential weight averaging

HYPOTHESIS: Averaging the final 20% of the optimization trajectory with decay 0.98 will exceed 9,267 correct predictions by reducing late-step parameter noise without changing the successful representation or inference ensemble.

INTENDED_EDIT: Track a bias-corrected EMA of all floating model parameters and normalization buffers during the final training phase, then install the averaged state after the last optimizer step.

EVIDENCE: Geometric-blend refinements have exhausted the float-level decision boundary without exceeding 9,267 correct, while wider-context architectural changes regressed or timed out; late weight averaging tests a distinct generalization mechanism while preserving the best architecture and TTA.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        weight_decay=2.0e-4,
        betas=(0.9, 0.99),
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        weight_decay=2.0e-4,
        betas=(0.9, 0.99),
    )
    optimizer._averaging_model = model
    optimizer._averaging_start = max(1, int(0.8 * total_steps))
    optimizer._averaging_decay = 0.98
    optimizer._averaging_mass = 0.0
    optimizer._averaged_state = {
        name: torch.zeros_like(tensor)
        for name, tensor in model.state_dict().items()
        if tensor.is_floating_point()
    }
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier

    completed_steps = step + 1
    if completed_steps >= optimizer._averaging_start:
        decay = optimizer._averaging_decay
        model_state = optimizer._averaging_model.state_dict()
        with torch.no_grad():
            for name, average in optimizer._averaged_state.items():
                average.mul_(decay).add_(
                    model_state[name].detach(),
                    alpha=1.0 - decay,
                )
        optimizer._averaging_mass = (
            decay * optimizer._averaging_mass + 1.0 - decay
        )

        if completed_steps >= total_steps:
            inverse_mass = 1.0 / optimizer._averaging_mass
            with torch.no_grad():
                for name, average in optimizer._averaged_state.items():
                    model_state[name].copy_(average, non_blocking=True)
                    model_state[name].mul_(inverse_mass)
>>>>>>> REPLACE