MECHANISM: Curriculum-synchronized EMA reset

HYPOTHESIS: Resetting the all-tensor EMA at the verified 19/32 augmentation transition will exceed 9,172 correct predictions by excluding residual broad-augmentation trajectory averages from the terminal inference-aligned model.

INTENDED_EDIT: Restore the best 19/32 curriculum, retain EMA for parameters and floating buffers, and reinitialize its averages immediately after the first terminal-phase optimizer step.

EVIDENCE: The all-tensor-EMA 19/32 design achieved the best completed result at 9,172 correct, while unaveraged BatchNorm buffers fell to 9,168; with EMA rate 0.005, roughly 4% of the pre-transition average survives to evaluation, motivating a targeted reset without sacrificing terminal smoothing.

<<<<<<< SEARCH
    optimizer._ema_started = False
    return optimizer
=======
    optimizer._ema_started = False
    optimizer._ema_curriculum_reset = False
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    if step * 64 < total_steps * 39:
=======
    if step * 32 < total_steps * 19:
>>>>>>> REPLACE

<<<<<<< SEARCH
        if not optimizer._ema_started:
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.copy_(tensor)
            optimizer._ema_started = True
        else:
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.lerp_(tensor, 0.005)
=======
        if not optimizer._ema_started:
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.copy_(tensor)
            optimizer._ema_started = True
        elif (
            not optimizer._ema_curriculum_reset
            and step * 32 >= total_steps * 19
        ):
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.copy_(tensor)
            optimizer._ema_curriculum_reset = True
        else:
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.lerp_(tensor, 0.005)
>>>>>>> REPLACE