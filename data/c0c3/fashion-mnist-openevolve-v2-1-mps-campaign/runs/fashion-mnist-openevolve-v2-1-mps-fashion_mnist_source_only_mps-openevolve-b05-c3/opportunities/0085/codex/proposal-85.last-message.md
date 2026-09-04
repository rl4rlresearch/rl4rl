MECHANISM: Faster curriculum-synchronized BatchNorm-state EMA

HYPOTHESIS: Retaining the validated 0.015 terminal EMA for learned parameters while increasing BatchNorm-buffer EMA to 0.03 will exceed 9,201 correct predictions by adapting normalization statistics faster to the cardinal-augmentation distribution without sacrificing parameter smoothing.

INTENDED_EDIT: Track whether each EMA tensor is a model buffer, then use terminal EMA rates of 0.015 for parameters and 0.03 for floating-point buffers; preserve the 0.005 pre-transition rate.

EVIDENCE: The all-tensor residual design reached 9,201 correct at terminal rate 0.015, whereas parameter-only EMA reached 9,168; this shows BatchNorm-state averaging is valuable and motivates independently reducing its post-transition lag.

<<<<<<< SEARCH
    optimizer._ema_tensors = [
        tensor
        for tensor in tuple(model.parameters()) + tuple(model.buffers())
        if tensor.is_floating_point()
    ]
    optimizer._ema_values = [
        tensor.detach().clone() for tensor in optimizer._ema_tensors
    ]
=======
    parameters = tuple(model.parameters())
    buffers = tuple(model.buffers())
    optimizer._ema_tensors = [
        tensor
        for tensor in parameters + buffers
        if tensor.is_floating_point()
    ]
    optimizer._ema_is_buffer = [
        False for tensor in parameters if tensor.is_floating_point()
    ] + [
        True for tensor in buffers if tensor.is_floating_point()
    ]
    optimizer._ema_values = [
        tensor.detach().clone() for tensor in optimizer._ema_tensors
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        else:
            ema_rate = 0.015 if step * 32 >= total_steps * 19 else 0.005
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.lerp_(tensor, ema_rate)
=======
        else:
            terminal_phase = step * 32 >= total_steps * 19
            for average, tensor, is_buffer in zip(
                optimizer._ema_values,
                optimizer._ema_tensors,
                optimizer._ema_is_buffer,
            ):
                if terminal_phase:
                    ema_rate = 0.03 if is_buffer else 0.015
                else:
                    ema_rate = 0.005
                average.lerp_(tensor, ema_rate)
>>>>>>> REPLACE