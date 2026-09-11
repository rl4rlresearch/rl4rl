MECHANISM: Tail-coupled BatchNorm state averaging

HYPOTHESIS: Averaging BatchNorm running statistics with the same recency-weighted tail ensemble as the trainable parameters will exceed 9,287 correct predictions by reducing parameter/statistic mismatch at evaluation.

INTENDED_EDIT: Preserve the current architecture, training, ten-view pooling, and recency-0.75 parameter averaging while also averaging every BatchNorm running mean and variance over the sampled tail iterates.

EVIDENCE: Recency exponents 0.5, 0.75, and 1.0 all plateaued at 9,287 correct, while later-biased variants fell to 9,286; the current code averages BatchNorm affine parameters but retains terminal running statistics, leaving an untested source of tail-ensemble mismatch.

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
        buffer
        for module in model.modules()
        if isinstance(module, nn.BatchNorm2d)
        for buffer in (module.running_mean, module.running_var)
        if buffer is not None
    ]
    optimizer.tail_average_buffer_values = [
        buffer.detach().clone()
        for buffer in optimizer.tail_average_buffers
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
            for average, buffer in zip(
                optimizer.tail_average_buffer_values,
                optimizer.tail_average_buffers,
            ):
                average.lerp_(buffer, average_weight)

            if step + 1 == total_steps:
                for parameter, average in zip(
                    parameters,
                    optimizer.tail_average_parameters,
                ):
                    parameter.copy_(average)
                for buffer, average in zip(
                    optimizer.tail_average_buffers,
                    optimizer.tail_average_buffer_values,
                ):
                    buffer.copy_(average)
>>>>>>> REPLACE