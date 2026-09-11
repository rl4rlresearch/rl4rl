MECHANISM: Decision-preserving TTA temperature sharpening

HYPOTHESIS: Applying 1.05× sharpening to the qualified split-EMA model’s ensemble logits will retain 9,206 correct predictions while lowering cross-entropy below 0.235889, improving validation_score.

INTENDED_EDIT: Restore the qualified 0.015 parameter/0.0325 BatchNorm-buffer terminal EMA rates and sharpen only evaluation-time ensemble logits without changing their argmax.

EVIDENCE: Reference Design 1 achieved 9,206 correct with 0.235889 cross-entropy, while the added moment branch fell to 9,201; fixed positive logit scaling preserves the stronger model’s predictions and directly targets the cross-entropy tie-breaker.

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log()
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer._ema_tensors = [
        tensor
        for tensor in tuple(model.parameters()) + tuple(model.buffers())
        if tensor.is_floating_point()
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
                    ema_rate = 0.0325 if is_buffer else 0.015
                else:
                    ema_rate = 0.005
                average.lerp_(tensor, ema_rate)
>>>>>>> REPLACE