MECHANISM: Tail-aligned BatchNorm buffer averaging

HYPOTHESIS: Averaging BatchNorm running statistics over the same tail checkpoints as the parameters will exceed 9,311 correct predictions by eliminating parameter/statistic misalignment.

INTENDED_EDIT: Retain the verified 0.5 tail-parameter blend while also averaging and blending all floating-point model buffers, including BatchNorm running means and variances.

EVIDENCE: The 0.5 tail blend achieved 9,311 correct versus 9,307 at 0.25 and 9,306 at 0.75, showing the existing midpoint is useful; prior buffer-alignment attempts timed out and therefore left this computation-light correction untested.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=8e-4, weight_decay=2e-4
    )
    optimizer._model = model
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        with torch.no_grad():
            if not hasattr(optimizer, "_tail_average"):
                optimizer._tail_average = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._tail_average_count = 1
            else:
                optimizer._tail_average_count += 1
                weight = 1.0 / optimizer._tail_average_count
                for average, parameter in zip(
                    optimizer._tail_average, parameters
                ):
                    average.lerp_(parameter.detach(), weight)

            if update == total_steps:
                for parameter, average in zip(
                    parameters, optimizer._tail_average
                ):
                    parameter.lerp_(average, 0.5)
=======
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        buffers = [
            buffer
            for buffer in optimizer._model.buffers()
            if buffer.is_floating_point()
        ]
        with torch.no_grad():
            if not hasattr(optimizer, "_tail_average"):
                optimizer._tail_average = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._tail_buffer_average = [
                    buffer.detach().clone() for buffer in buffers
                ]
                optimizer._tail_average_count = 1
            else:
                optimizer._tail_average_count += 1
                weight = 1.0 / optimizer._tail_average_count
                for average, parameter in zip(
                    optimizer._tail_average, parameters
                ):
                    average.lerp_(parameter.detach(), weight)
                for average, buffer in zip(
                    optimizer._tail_buffer_average, buffers
                ):
                    average.lerp_(buffer.detach(), weight)

            if update == total_steps:
                for parameter, average in zip(
                    parameters, optimizer._tail_average
                ):
                    parameter.lerp_(average, 0.5)
                for buffer, average in zip(
                    buffers, optimizer._tail_buffer_average
                ):
                    buffer.lerp_(average, 0.5)
>>>>>>> REPLACE