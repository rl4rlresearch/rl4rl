MECHANISM: BatchNorm-aligned tail checkpoint averaging

HYPOTHESIS: Averaging BatchNorm running statistics alongside the tail-averaged parameters will exceed 9,310 correct validation predictions by eliminating the state mismatch between averaged weights and final-iterate normalization buffers.

INTENDED_EDIT: Track floating-point BatchNorm buffers at each existing tail sample and blend their averages into the final model using the same 0.5 interpolation as learned parameters.

EVIDENCE: The normalized 147-unit head achieved the best result of 9,310 correct, and reducing parameter averaging to 0.25 regressed to 9,307; the prior BatchNorm-alignment verification timed out, so its accuracy effect remains untested while requiring negligible extra computation.

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
    optimizer._normalization_buffers = [
        buffer
        for module in model.modules()
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
        for buffer in (module.running_mean, module.running_var)
    ]
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
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
            buffers = optimizer._normalization_buffers
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