MECHANISM: Faster post-transition BatchNorm-state tracking

HYPOTHESIS: Using a 0.015 terminal EMA rate for learned parameters and 0.04 for BatchNorm buffers will exceed 9,206 correct predictions by further reducing normalization-state lag while preserving the validated parameter smoothing.

INTENDED_EDIT: Mark EMA tensors that are model buffers, then apply terminal EMA rates of 0.015 to parameters and 0.04 to buffers while retaining the 0.005 pre-transition rate.

EVIDENCE: Separating the terminal rates at 0.015 for parameters and 0.03 for buffers reported 9,206 correct versus 9,201 when all tensors used 0.015, indicating that faster BatchNorm-buffer adaptation is the promising axis to continue.

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
    buffer_ids = {id(tensor) for tensor in buffers}
    optimizer._ema_is_buffer = [
        id(tensor) in buffer_ids for tensor in optimizer._ema_tensors
    ]
    optimizer._ema_values = [
        tensor.detach().clone() for tensor in optimizer._ema_tensors
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        else:
            ema_rate = 0.01 if step * 32 >= total_steps * 19 else 0.005
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
                    ema_rate = 0.04 if is_buffer else 0.015
                else:
                    ema_rate = 0.005
                average.lerp_(tensor, ema_rate)
>>>>>>> REPLACE