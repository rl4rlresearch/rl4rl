MECHANISM: Tail exponential weight averaging

HYPOTHESIS: Averaging parameters and BatchNorm statistics over the final half of training will exceed 9,257 correct predictions by reducing endpoint noise from the cosine-decayed AdamW trajectory without changing the proven architecture or example exposure.

INTENDED_EDIT: Track a 0.99-decay exponential moving average during the final 50% of optimizer steps and replace the model’s parameters and floating-point buffers with that average after the last step.

EVIDENCE: The 233,434-parameter architecture remains best, while further widening, dropout, pooling, global readouts, alternative aggregation, rotation, and smaller-batch optimization all regressed; tail weight averaging preserves that design and tests optimizer-trajectory stability, an unexplored axis.

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
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        weight_decay=2.0e-4,
        betas=(0.9, 0.99),
    )
    optimizer._ema_model = model
    optimizer._ema_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    optimizer._ema_buffers = [
        (buffer, buffer.detach().clone())
        for buffer in model.buffers()
        if buffer.is_floating_point()
    ]
    optimizer._ema_started = False
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier

    if progress >= 0.5:
        decay = 0.99
        with torch.no_grad():
            if not optimizer._ema_started:
                for parameter, average in zip(
                    optimizer._ema_model.parameters(),
                    optimizer._ema_parameters,
                ):
                    average.copy_(parameter)
                for buffer, average in optimizer._ema_buffers:
                    average.copy_(buffer)
                optimizer._ema_started = True
            else:
                for parameter, average in zip(
                    optimizer._ema_model.parameters(),
                    optimizer._ema_parameters,
                ):
                    average.lerp_(parameter, 1.0 - decay)
                for buffer, average in optimizer._ema_buffers:
                    average.lerp_(buffer, 1.0 - decay)

            if step + 1 >= total_steps:
                for parameter, average in zip(
                    optimizer._ema_model.parameters(),
                    optimizer._ema_parameters,
                ):
                    parameter.copy_(average)
                for buffer, average in optimizer._ema_buffers:
                    buffer.copy_(average)
>>>>>>> REPLACE