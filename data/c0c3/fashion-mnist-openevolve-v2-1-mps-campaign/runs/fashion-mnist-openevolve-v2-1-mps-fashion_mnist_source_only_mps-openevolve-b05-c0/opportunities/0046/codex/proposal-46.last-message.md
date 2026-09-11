MECHANISM: BatchNorm-buffer-aware tail EMA

HYPOTHESIS: Averaging BatchNorm running statistics alongside the proven full-parameter tail EMA will exceed 9,254 correct predictions by aligning evaluation normalization with the averaged convolutional weights.

INTENDED_EDIT: Retain access to the model through the optimizer, EMA all floating-point model buffers on the existing stride, and restore them with the averaged parameters after training.

EVIDENCE: Full-parameter EMA improved correct predictions from 9,250 to 9,254, while the classifier-only EMA attempt specifically identified possible misalignment between averaged feature parameters and final BatchNorm statistics; averaging the small BatchNorm buffers directly tests that mechanism with negligible added work.

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
    optimizer._model = model
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
        floating_buffers = [
            buffer
            for buffer in optimizer._model.buffers()
            if buffer.is_floating_point()
        ]
        ema_parameters = getattr(optimizer, "_ema_parameters", None)
        ema_buffers = getattr(optimizer, "_ema_buffers", None)
        with torch.no_grad():
            ema_start = total_steps // 2
            if ema_parameters is None:
                ema_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
                ema_buffers = [
                    buffer.detach().clone() for buffer in floating_buffers
                ]
                optimizer._ema_parameters = ema_parameters
                optimizer._ema_buffers = ema_buffers
            elif (completed_steps - ema_start) % 4 == 0:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.01)
                for average, buffer in zip(ema_buffers, floating_buffers):
                    average.lerp_(buffer.detach(), 0.01)

            if completed_steps >= total_steps:
                for parameter, average in zip(parameters, ema_parameters):
                    parameter.copy_(average)
                for buffer, average in zip(floating_buffers, ema_buffers):
                    buffer.copy_(average)
>>>>>>> REPLACE