MECHANISM: EMA-aligned batch-normalization statistics

HYPOTHESIS: Averaging floating-point normalization buffers alongside the EMA weights will produce a validation_score above 9360.421693447506 by reducing evaluation-statistic noise or improving at least one classification decision.

INTENDED_EDIT: Update and install EMA copies of all floating-point model buffers using the same decay and lifecycle as the parameter EMA.

EVIDENCE: Agreement calibration repeatedly preserved exactly 9,360 predictions and is now saturated, while the current implementation replaces parameters with EMA values but leaves its already-collected `ema_buffer_sources` unused, creating a mismatch between averaged weights and current BatchNorm statistics.

<<<<<<< SEARCH
        if optimizer.ema_pairs is None:
            optimizer.ema_pairs = [
                (parameter, parameter.detach().clone())
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
        else:
            updates = step + 1
            ema_decay = min(0.98, (updates + 1.0) / (updates + 10.0))
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
            ema_decay = min(0.98, (updates + 1.0) / (updates + 10.0))
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