MECHANISM: EMA-consistent BatchNorm state

HYPOTHESIS: Averaging BatchNorm running statistics with the same decay as the EMA parameters will exceed 9,360 correct predictions by eliminating the final mismatch between averaged weights and non-averaged normalization state.

INTENDED_EDIT: Track floating-point model buffers during EMA updates and restore their averages alongside parameter averages at the final optimizer step.

EVIDENCE: The widened spatial head reduced accuracy from 9,360 to 9,326, favoring preservation of the current architecture. The current procedure replaces every learned parameter with its EMA value but leaves BatchNorm running statistics from the raw training trajectory, despite already collecting the relevant buffers.

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