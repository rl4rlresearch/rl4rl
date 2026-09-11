MECHANISM: BatchNorm-consistent tail EMA

HYPOTHESIS: Averaging and restoring BatchNorm running statistics alongside the strided parameter EMA will exceed 9,254 correct predictions by evaluating the averaged weights with normalization statistics from the same averaging window.

INTENDED_EDIT: Track the three BatchNorm layers’ running means and variances, update their EMAs whenever the parameter EMA updates, and restore both together after training.

EVIDENCE: BatchNorm raised validation correct from 9,170 to 9,210, while strided tail EMA raised the batch-32 result from 9,250 to 9,254; the current implementation averages weights but leaves BatchNorm statistics tied to the unaveraged training trajectory.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2.5e-3, weight_decay=1e-4
    )
    optimizer._normalization_buffers = [
        buffer
        for module in model.modules()
        if isinstance(module, nn.BatchNorm2d)
        for buffer in (module.running_mean, module.running_var)
        if buffer is not None
    ]
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
        ema_parameters = getattr(optimizer, "_ema_parameters", None)
        with torch.no_grad():
            ema_start = total_steps // 2
            if ema_parameters is None:
                ema_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._ema_parameters = ema_parameters
            elif (completed_steps - ema_start) % 4 == 0:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.01)

            if completed_steps >= total_steps:
                for parameter, average in zip(parameters, ema_parameters):
                    parameter.copy_(average)
=======
        ema_parameters = getattr(optimizer, "_ema_parameters", None)
        normalization_buffers = optimizer._normalization_buffers
        ema_buffers = getattr(optimizer, "_ema_buffers", None)
        with torch.no_grad():
            ema_start = total_steps // 2
            if ema_parameters is None:
                ema_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
                ema_buffers = [
                    buffer.detach().clone() for buffer in normalization_buffers
                ]
                optimizer._ema_parameters = ema_parameters
                optimizer._ema_buffers = ema_buffers
            elif (completed_steps - ema_start) % 4 == 0:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.01)
                for average, buffer in zip(ema_buffers, normalization_buffers):
                    average.lerp_(buffer.detach(), 0.01)

            if completed_steps >= total_steps:
                for parameter, average in zip(parameters, ema_parameters):
                    parameter.copy_(average)
                for buffer, average in zip(normalization_buffers, ema_buffers):
                    buffer.copy_(average)
>>>>>>> REPLACE