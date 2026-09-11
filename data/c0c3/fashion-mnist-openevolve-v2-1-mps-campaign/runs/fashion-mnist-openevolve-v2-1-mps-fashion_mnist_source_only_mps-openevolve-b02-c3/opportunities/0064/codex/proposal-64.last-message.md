MECHANISM: Mid-horizon full-state exponential weight consolidation

HYPOTHESIS: A 0.9825-decay full-state EMA will exceed 9,344 correct predictions by interpolating between the near-best 0.985 horizon and the best 0.98 horizon.

INTENDED_EDIT: Restore the proven pooled mean/max refinement gate and add foreach-accelerated averaging of all floating-point parameters and BatchNorm buffers from halfway through training, using decay 0.9825.

EVIDENCE: Full-state EMA improved from 9,334 correct at decay 0.99 to 9,343 at 0.985 and peaked at 9,344 at 0.98 before regressing to 9,340 at 0.975, motivating a focused midpoint test in the bracket containing the observed optimum.

<<<<<<< SEARCH
        mean_descriptor = F.adaptive_avg_pool2d(refined, output_size=1)
        peak_descriptor = F.adaptive_max_pool2d(refined, output_size=1)
        gate_features = 0.5 * (
            F.gelu(self.gate_down(mean_descriptor))
            + F.gelu(self.gate_down(peak_descriptor))
        )
        gate = 2.0 * torch.sigmoid(self.gate_up(gate_features))
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
    return torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=5e-4,
    )
=======
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
                decay = 0.9825 ** elapsed
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