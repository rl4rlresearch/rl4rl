MECHANISM: Endpoint-interpolated BatchNorm statistics

HYPOTHESIS: Approximating the proven tail-EMA BatchNorm statistics from their midpoint and final values will retain at least 9,255 correct predictions without the runtime cost of updating buffer averages throughout training.

INTENDED_EDIT: Attach the model to the optimizer, snapshot floating-point buffers when parameter EMA begins, and interpolate BatchNorm statistics to the EMA’s approximate 75%-through-tail position when restoring averaged parameters.

EVIDENCE: Full BatchNorm-buffer EMA improved validation_correct from 9,254 to 9,255 but took 84.95 seconds; its parameter EMA effectively represents roughly 75% of the way through the tail, so one endpoint interpolation targets the same alignment with negligible per-step work.

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
            if ema_parameters is None:
                ema_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._ema_parameters = ema_parameters
                optimizer._ema_buffer_starts = [
                    buffer.detach().clone()
                    for buffer in optimizer._model.buffers()
                    if torch.is_floating_point(buffer)
                ]
            elif (completed_steps - ema_start) % 4 == 0:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.01)

            if completed_steps >= total_steps:
                for parameter, average in zip(parameters, ema_parameters):
                    parameter.copy_(average)
                buffers = [
                    buffer
                    for buffer in optimizer._model.buffers()
                    if torch.is_floating_point(buffer)
                ]
                for buffer, start in zip(
                    buffers, optimizer._ema_buffer_starts
                ):
                    buffer.lerp_(start, 0.25)
>>>>>>> REPLACE