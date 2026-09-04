MECHANISM: BatchNorm-aligned tail averaging

HYPOTHESIS: Synchronizing BatchNorm running statistics with the existing tail-averaged parameters will achieve at least 9,311 correct validation predictions without materially increasing runtime.

INTENDED_EDIT: Track floating-point BatchNorm buffers during tail sampling and blend their averages into the final model with the same 0.5 interpolation used for learned parameters.

EVIDENCE: The 249,601-parameter normalized head reached 9,310 correct, and reducing tail interpolation to 0.25 fell to 9,307, showing that averaging helps; however, the current procedure averages BatchNorm parameters without their running statistics, leaving an avoidable state mismatch.

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
    optimizer._batch_norm_buffers = [
        buffer
        for module in model.modules()
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
        for buffer in (module.running_mean, module.running_var)
        if buffer is not None
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
            buffers = optimizer._batch_norm_buffers
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