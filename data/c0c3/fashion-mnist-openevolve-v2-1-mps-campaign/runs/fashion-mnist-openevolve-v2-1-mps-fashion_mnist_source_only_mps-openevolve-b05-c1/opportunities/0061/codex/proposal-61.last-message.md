MECHANISM: BatchNorm-statistics-aware exponential weight averaging

HYPOTHESIS: Averaging BatchNorm running means and variances with the same trajectory as the learned parameters will exceed 9,342 correct predictions by eliminating the evaluation mismatch between EMA weights and final-step normalization statistics.

INTENDED_EDIT: Extend the existing EMA to track and install every floating-point model buffer alongside learned parameters, without changing architecture, parameter count, or forward-pass cost.

EVIDENCE: Parameter-only EMA improved correctness from 9,334 to 9,342, establishing that trajectory averaging helps; the current implementation leaves BatchNorm buffers at their final-step values, making matched normalization statistics the most direct low-cost refinement.

<<<<<<< SEARCH
    optimizer.ema_pairs = None
    return optimizer
=======
    optimizer.ema_pairs = None
    optimizer.ema_buffer_sources = [
        buffer
        for buffer in model.buffers()
        if torch.is_floating_point(buffer)
    ]
    optimizer.ema_buffer_pairs = None
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
        if optimizer.ema_pairs is None:
            optimizer.ema_pairs = [
                (parameter, parameter.detach().clone())
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
        else:
            updates = step + 1
            ema_decay = min(0.99, (updates + 1.0) / (updates + 10.0))
            for parameter, average in optimizer.ema_pairs:
                average.lerp_(parameter, 1.0 - ema_decay)

        if step + 1 >= total_steps:
            for parameter, average in optimizer.ema_pairs:
                parameter.copy_(average)
=======
        if optimizer.ema_pairs is None:
            optimizer.ema_pairs = [
                (parameter, parameter.detach().clone())
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            optimizer.ema_buffer_pairs = [
                (buffer, buffer.detach().clone())
                for buffer in optimizer.ema_buffer_sources
            ]
        else:
            updates = step + 1
            ema_decay = min(0.99, (updates + 1.0) / (updates + 10.0))
            for parameter, average in optimizer.ema_pairs:
                average.lerp_(parameter, 1.0 - ema_decay)
            for buffer, average in optimizer.ema_buffer_pairs:
                average.lerp_(buffer, 1.0 - ema_decay)

        if step + 1 >= total_steps:
            for parameter, average in optimizer.ema_pairs:
                parameter.copy_(average)
            for buffer, average in optimizer.ema_buffer_pairs:
                buffer.copy_(average)
>>>>>>> REPLACE