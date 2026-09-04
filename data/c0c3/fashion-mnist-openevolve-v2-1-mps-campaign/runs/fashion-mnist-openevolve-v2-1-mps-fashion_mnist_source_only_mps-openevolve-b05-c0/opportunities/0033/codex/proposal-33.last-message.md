MECHANISM: BatchNorm-consistent parameter EMA

HYPOTHESIS: Averaging BatchNorm running statistics alongside the proven tail parameter EMA will exceed 9,254 correct predictions by eliminating the mismatch between averaged weights and final-step normalization statistics.

INTENDED_EDIT: Retain the model on the optimizer and apply the existing strided EMA to floating-point model buffers, restoring both averaged parameters and BatchNorm statistics before validation.

EVIDENCE: BatchNorm previously improved correct predictions from 9,170 to 9,210, and the parameter EMA improved the batch-32 result from 9,250 to 9,254; these results motivate making the two beneficial mechanisms internally consistent.

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
    optimizer._training_model = model
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
        model = optimizer._training_model
        buffers = [
            buffer for buffer in model.buffers() if buffer.is_floating_point()
        ]
        ema_parameters = getattr(optimizer, "_ema_parameters", None)
        ema_buffers = getattr(optimizer, "_ema_buffers", None)
        with torch.no_grad():
            ema_start = total_steps // 2
            if ema_parameters is None:
                ema_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
                ema_buffers = [buffer.detach().clone() for buffer in buffers]
                optimizer._ema_parameters = ema_parameters
                optimizer._ema_buffers = ema_buffers
            elif (completed_steps - ema_start) % 4 == 0:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.01)
                for average, buffer in zip(ema_buffers, buffers):
                    average.lerp_(buffer.detach(), 0.01)

            if completed_steps >= total_steps:
                for parameter, average in zip(parameters, ema_parameters):
                    parameter.copy_(average)
                for buffer, average in zip(buffers, ema_buffers):
                    buffer.copy_(average)
>>>>>>> REPLACE