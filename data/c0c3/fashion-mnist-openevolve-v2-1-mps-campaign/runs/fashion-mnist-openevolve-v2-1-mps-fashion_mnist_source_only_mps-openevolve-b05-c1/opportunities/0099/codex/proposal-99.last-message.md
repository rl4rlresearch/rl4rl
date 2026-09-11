MECHANISM: Live BatchNorm calibration with EMA weights

HYPOTHESIS: Retaining the final live BatchNorm statistics when installing EMA parameters will exceed 9,345 correct predictions by avoiding compounded temporal lag in the activation statistics.

INTENDED_EDIT: Continue averaging learned parameters, but stop averaging and replacing BatchNorm buffers at the end of training.

EVIDENCE: The 9,345-correct baseline averages both parameters and already-smoothed BatchNorm running statistics, while dropout and label-smoothing variants failed to improve it; correcting this orthogonal train-to-evaluation state mismatch adds no parameters or material runtime.

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE