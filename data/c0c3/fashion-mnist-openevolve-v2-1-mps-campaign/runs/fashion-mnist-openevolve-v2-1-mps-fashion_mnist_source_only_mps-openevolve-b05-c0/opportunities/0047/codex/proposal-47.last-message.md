MECHANISM: Fused BatchNorm-aware tail EMA

HYPOTHESIS: Averaging BatchNorm statistics with the tail-EMA parameters will reach at least 9,255 correct predictions, while fused tensor updates will reduce the 84.95-second runtime enough to finish verification.

INTENDED_EDIT: Retain the model on the optimizer, include floating-point buffers in EMA and restoration, and fuse each EMA update into one foreach operation.

EVIDENCE: BatchNorm-buffer-aware EMA produced 9,255 correct versus the current 9,254, but took 84.95 seconds; fusing its many per-tensor EMA operations targets that observed runtime regression without changing the averaging schedule.

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
    if completed_steps >= total_steps // 2:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
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
    if completed_steps >= total_steps // 2:
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
        current_tensors = parameters + buffers
        ema_tensors = getattr(optimizer, "_ema_tensors", None)
        with torch.no_grad():
            ema_start = total_steps // 2
            if ema_tensors is None:
                ema_tensors = [
                    tensor.detach().clone() for tensor in current_tensors
                ]
                optimizer._ema_tensors = ema_tensors
            elif (completed_steps - ema_start) % 4 == 0:
                torch._foreach_lerp_(
                    ema_tensors,
                    [tensor.detach() for tensor in current_tensors],
                    0.01,
                )

            if completed_steps >= total_steps:
                torch._foreach_copy_(current_tensors, ema_tensors)
>>>>>>> REPLACE