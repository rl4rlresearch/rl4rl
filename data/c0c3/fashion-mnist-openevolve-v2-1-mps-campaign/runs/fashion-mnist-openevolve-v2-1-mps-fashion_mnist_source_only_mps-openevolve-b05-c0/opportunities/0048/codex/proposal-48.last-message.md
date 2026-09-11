MECHANISM: Sparse BatchNorm-statistics tail EMA

HYPOTHESIS: Averaging BatchNorm running statistics every 32 steps with a decay equivalent to the proven four-step EMA will retain at least 9,255 correct predictions while avoiding its observed 84.95-second runtime regression.

INTENDED_EDIT: Expose floating-point model buffers to the optimizer, sparsely EMA them during the existing parameter-EMA tail, and restore both averaged parameters and statistics for evaluation.

EVIDENCE: BatchNorm-buffer-aware EMA improved validation_correct from 9,254 to 9,255 but took 84.95 seconds; reducing buffer updates eightfold directly preserves that winning mechanism while targeting its runtime failure.

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
    optimizer._ema_buffer_sources = [
        buffer for buffer in model.buffers() if torch.is_floating_point(buffer)
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
        buffer_sources = optimizer._ema_buffer_sources
        ema_buffers = getattr(optimizer, "_ema_buffers", None)
        with torch.no_grad():
            ema_start = total_steps // 2
            if ema_parameters is None:
                ema_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._ema_parameters = ema_parameters
                ema_buffers = [
                    buffer.detach().clone() for buffer in buffer_sources
                ]
                optimizer._ema_buffers = ema_buffers
            elif (completed_steps - ema_start) % 4 == 0:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.01)

                if (completed_steps - ema_start) % 32 == 0:
                    torch._foreach_lerp_(
                        ema_buffers,
                        buffer_sources,
                        1.0 - 0.99**8,
                    )

            if completed_steps >= total_steps:
                for parameter, average in zip(parameters, ema_parameters):
                    parameter.copy_(average)
                for buffer, average in zip(buffer_sources, ema_buffers):
                    buffer.copy_(average)
>>>>>>> REPLACE