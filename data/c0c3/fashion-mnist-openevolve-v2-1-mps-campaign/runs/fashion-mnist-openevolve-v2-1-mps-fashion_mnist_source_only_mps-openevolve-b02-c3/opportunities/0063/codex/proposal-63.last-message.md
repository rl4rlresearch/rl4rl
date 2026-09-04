MECHANISM: Short-horizon full-state exponential weight consolidation

HYPOTHESIS: A 0.975-decay full-state EMA will exceed 9,344 correct predictions by tracking final low-learning-rate convergence more closely than the successful 0.98 EMA.

INTENDED_EDIT: Restore the proven equal mean/max refinement descriptor and foreach-accelerated full-state EMA, shortening its decay from 0.98 to 0.975.

EVIDENCE: Shortening full-state EMA decay from 0.99 to 0.985 improved correctness from 9,334 to 9,343, and 0.98 further improved it to 9,344; this motivates one localized continuation of the successful trend.

<<<<<<< SEARCH
        pooled = F.adaptive_avg_pool2d(refined, output_size=1)
        gate = 2.0 * torch.sigmoid(
            self.gate_up(F.gelu(self.gate_down(pooled)))
        )
=======
        pooled = 0.5 * (
            F.adaptive_avg_pool2d(refined, output_size=1)
            + F.adaptive_max_pool2d(refined, output_size=1)
        )
        gate = 2.0 * torch.sigmoid(
            self.gate_up(F.gelu(self.gate_down(pooled)))
        )
>>>>>>> REPLACE

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
    optimizer._averaging_values = tuple(
        value
        for value in model.state_dict().values()
        if torch.is_floating_point(value)
    )
    optimizer._averaged_values = None
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
        values = optimizer._averaging_values
        averaged_values = optimizer._averaged_values
        last_step = optimizer._averaging_last_step
        with torch.no_grad():
            if averaged_values is None:
                averaged_values = [
                    value.detach().clone() for value in values
                ]
            else:
                elapsed = max(step - last_step, 1)
                decay = 0.975 ** elapsed
                torch._foreach_mul_(averaged_values, decay)
                torch._foreach_add_(
                    averaged_values,
                    values,
                    alpha=1.0 - decay,
                )
        optimizer._averaged_values = averaged_values
        optimizer._averaging_last_step = step

        if step >= total_steps:
            with torch.no_grad():
                torch._foreach_copy_(values, averaged_values)
>>>>>>> REPLACE