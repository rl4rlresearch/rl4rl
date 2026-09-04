MECHANISM: EMA-consistent BatchNorm statistics

HYPOTHESIS: Blending BatchNorm running statistics toward their bias-corrected EMA by the same factor as learned parameters will exceed 9,166 correct predictions by removing the state mismatch in the final averaged model.

INTENDED_EDIT: Track EMAs of all BatchNorm running means and variances during training, then apply the existing final interpolation factor to those buffers alongside the parameters.

EVIDENCE: The 9,166-correct baseline remains unbeaten by classifier, pooling, loss, augmentation, and TTA changes; this preserves that implementation while addressing its currently unaveraged BatchNorm state at negligible computational cost.

<<<<<<< SEARCH
    optimizer._ema_shadow = [
        torch.zeros_like(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    optimizer._ema_updates = 0
=======
    optimizer._ema_shadow = [
        torch.zeros_like(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    optimizer._ema_buffers = [
        (
            buffer,
            torch.zeros_like(buffer),
        )
        for name, buffer in model.named_buffers()
        if name.endswith("running_mean") or name.endswith("running_var")
    ]
    optimizer._ema_updates = 0
>>>>>>> REPLACE

<<<<<<< SEARCH
    with torch.no_grad():
        optimizer._ema_updates += 1
        for shadow, parameter in zip(optimizer._ema_shadow, parameters):
            shadow.mul_(ema_decay).add_(parameter, alpha=1.0 - ema_decay)
        if step + 1 >= total_steps:
            correction = 1.0 - ema_decay ** optimizer._ema_updates
            for shadow, parameter in zip(optimizer._ema_shadow, parameters):
                parameter.lerp_(shadow / correction, 0.2861344814300537109375)
=======
    with torch.no_grad():
        optimizer._ema_updates += 1
        for shadow, parameter in zip(optimizer._ema_shadow, parameters):
            shadow.mul_(ema_decay).add_(parameter, alpha=1.0 - ema_decay)
        for buffer, shadow in optimizer._ema_buffers:
            shadow.mul_(ema_decay).add_(buffer, alpha=1.0 - ema_decay)
        if step + 1 >= total_steps:
            correction = 1.0 - ema_decay ** optimizer._ema_updates
            blend = 0.2861344814300537109375
            for shadow, parameter in zip(optimizer._ema_shadow, parameters):
                parameter.lerp_(shadow / correction, blend)
            for buffer, shadow in optimizer._ema_buffers:
                buffer.lerp_(shadow / correction, blend)
>>>>>>> REPLACE