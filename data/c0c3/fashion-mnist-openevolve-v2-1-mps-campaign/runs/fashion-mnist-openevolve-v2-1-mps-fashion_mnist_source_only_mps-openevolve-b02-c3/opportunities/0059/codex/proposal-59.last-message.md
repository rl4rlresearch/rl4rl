MECHANISM: Linearly interpolated sparse full-state EMA

HYPOTHESIS: Approximating the successful per-step 0.99 EMA by integrating a linear parameter trajectory between eight-step snapshots will exceed 9,334 correct predictions without incurring the per-step experiment’s timeout.

INTENDED_EDIT: Restore the equal mean/maximum gate, then average parameters and floating-point BatchNorm buffers using exponentially weighted interpolation between consecutive sparse snapshots.

EVIDENCE: Sparse full-state EMA improved the equal-statistic gated model from 9,322 to 9,334 correct, while parameter-only EMA reached 9,331 and per-step EMA timed out; this tests a more faithful approximation at the proven snapshot frequency.

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
    optimizer._averaging_model = model
    optimizer._averaged_tensors = None
    optimizer._averaging_previous_tensors = None
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
        model = optimizer._averaging_model
        state_tensors = tuple(model.parameters()) + tuple(
            buffer
            for buffer in model.buffers()
            if torch.is_floating_point(buffer)
        )
        averaged_tensors = optimizer._averaged_tensors
        previous_tensors = optimizer._averaging_previous_tensors
        last_step = optimizer._averaging_last_step

        with torch.no_grad():
            if averaged_tensors is None:
                averaged_tensors = [
                    value.detach().clone()
                    for value in state_tensors
                ]
                previous_tensors = [
                    value.detach().clone()
                    for value in state_tensors
                ]
            else:
                elapsed = max(step - last_step, 1)
                per_step_decay = 0.99
                decay = per_step_decay ** elapsed
                current_weight = sum(
                    (1.0 - per_step_decay)
                    * per_step_decay ** (elapsed - offset)
                    * offset
                    / elapsed
                    for offset in range(1, elapsed + 1)
                )
                previous_weight = (
                    1.0 - decay - current_weight
                )
                torch._foreach_mul_(averaged_tensors, decay)
                torch._foreach_add_(
                    averaged_tensors,
                    previous_tensors,
                    alpha=previous_weight,
                )
                torch._foreach_add_(
                    averaged_tensors,
                    state_tensors,
                    alpha=current_weight,
                )
                torch._foreach_copy_(
                    previous_tensors,
                    state_tensors,
                )

        optimizer._averaged_tensors = averaged_tensors
        optimizer._averaging_previous_tensors = previous_tensors
        optimizer._averaging_last_step = step

        if step >= total_steps:
            with torch.no_grad():
                torch._foreach_copy_(
                    state_tensors,
                    averaged_tensors,
                )
>>>>>>> REPLACE