MECHANISM: Short-horizon sparse exponential weight consolidation

HYPOTHESIS: A 0.985-decay sparse EMA on the proven mean/max-gated model will exceed 9,334 correct predictions by retaining the variance reduction of the successful 0.99 EMA while tracking the final low-learning-rate solution more closely.

INTENDED_EDIT: Restore Reference Design 2’s equal mean/max descriptor and sparse late-training EMA, but shorten the EMA memory from roughly 100 to 67 optimizer steps.

EVIDENCE: Reference Design 2’s 0.99-decay sparse EMA achieved the best result at 9,334 correct, improving the same unaveraged architecture’s 9,322; this motivates localized tuning of the successful averaging horizon rather than another architectural change.

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
    optimizer._averaging_model = model
    optimizer._averaged_state = {}
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
        averaged_state = optimizer._averaged_state
        last_step = optimizer._averaging_last_step
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if not torch.is_floating_point(value):
                    continue
                if name not in averaged_state:
                    averaged_state[name] = value.detach().clone()
                else:
                    elapsed = max(step - last_step, 1)
                    decay = 0.985 ** elapsed
                    averaged_state[name].mul_(decay).add_(
                        value.detach(),
                        alpha=1.0 - decay,
                    )
        optimizer._averaging_last_step = step

        if step >= total_steps:
            with torch.no_grad():
                for name, value in model.state_dict().items():
                    if name in averaged_state:
                        value.copy_(averaged_state[name])
>>>>>>> REPLACE