MECHANISM: Tail-aligned BatchNorm buffer averaging

HYPOTHESIS: Linearly averaging BatchNorm running statistics over the same final-10% window as learned parameters will exceed 9,285 correct predictions by eliminating the mismatch between averaged weights and terminal normalization statistics.

INTENDED_EDIT: Track floating-point model buffers and apply the existing linear-recency tail average to them, adding negligible computation and no learned parameters.

EVIDENCE: Linear-recency parameter averaging achieved the best verified result of 9,285 correct versus 9,282 for uniform averaging; the current design still pairs those averaged parameters with terminal BatchNorm statistics, making normalization-state alignment the most targeted next test.

<<<<<<< SEARCH
    optimizer.tail_average_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return optimizer
=======
    optimizer.tail_average_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer.tail_average_buffers = [
        (buffer, buffer.detach().clone())
        for buffer in model.buffers()
        if buffer.is_floating_point()
    ]
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
        with torch.no_grad():
            for average, parameter in zip(
                optimizer.tail_average_parameters,
                parameters,
            ):
                average.lerp_(parameter, average_weight)

            if step + 1 == total_steps:
                for parameter, average in zip(
                    parameters,
                    optimizer.tail_average_parameters,
                ):
                    parameter.copy_(average)
=======
        with torch.no_grad():
            for average, parameter in zip(
                optimizer.tail_average_parameters,
                parameters,
            ):
                average.lerp_(parameter, average_weight)
            for buffer, average in optimizer.tail_average_buffers:
                average.lerp_(buffer, average_weight)

            if step + 1 == total_steps:
                for parameter, average in zip(
                    parameters,
                    optimizer.tail_average_parameters,
                ):
                    parameter.copy_(average)
                for buffer, average in optimizer.tail_average_buffers:
                    buffer.copy_(average)
>>>>>>> REPLACE