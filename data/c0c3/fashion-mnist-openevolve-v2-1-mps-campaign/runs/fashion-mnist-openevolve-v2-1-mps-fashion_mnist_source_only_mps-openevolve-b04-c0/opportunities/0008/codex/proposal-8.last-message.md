MECHANISM: EMA-consistent BatchNorm statistics

HYPOTHESIS: Averaging BatchNorm running means and variances alongside the already successful parameter EMA will exceed 9,323 correct predictions by eliminating the evaluation mismatch between averaged weights and final-step normalization statistics.

INTENDED_EDIT: Extend the second-half 0.99-decay EMA to floating-point model buffers and install those averaged buffers with the averaged parameters at the final step.

EVIDENCE: Parameter EMA improved validation_correct from 9,322 to 9,323 and cross-entropy from 0.19535 to 0.19251; the current implementation averages BatchNorm parameters but leaves their running statistics from the unaveraged final model, motivating a coherence fix without altering exposure or architecture.

<<<<<<< SEARCH
    ema_start = max(total_steps // 2, 1)
    ema_parameters: list[torch.Tensor] = []
    optimizer_step = 0

    def update_ema(
        _: torch.optim.Optimizer,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        del args, kwargs
        nonlocal ema_parameters, optimizer_step
        optimizer_step += 1
        with torch.no_grad():
            if optimizer_step == ema_start:
                ema_parameters = [
                    parameter.detach().clone()
                    for parameter in model.parameters()
                ]
            elif optimizer_step > ema_start:
                for average, parameter in zip(
                    ema_parameters, model.parameters()
                ):
                    average.lerp_(parameter.detach(), 0.01)

            if optimizer_step == total_steps:
                for parameter, average in zip(
                    model.parameters(), ema_parameters
                ):
                    parameter.copy_(average)
=======
    ema_start = max(total_steps // 2, 1)
    ema_parameters: list[torch.Tensor] = []
    ema_buffers: list[torch.Tensor] = []
    optimizer_step = 0

    def update_ema(
        _: torch.optim.Optimizer,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        del args, kwargs
        nonlocal ema_parameters, ema_buffers, optimizer_step
        optimizer_step += 1
        floating_buffers = [
            buffer for buffer in model.buffers()
            if buffer.is_floating_point()
        ]
        with torch.no_grad():
            if optimizer_step == ema_start:
                ema_parameters = [
                    parameter.detach().clone()
                    for parameter in model.parameters()
                ]
                ema_buffers = [
                    buffer.detach().clone()
                    for buffer in floating_buffers
                ]
            elif optimizer_step > ema_start:
                for average, parameter in zip(
                    ema_parameters, model.parameters()
                ):
                    average.lerp_(parameter.detach(), 0.01)
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.01)

            if optimizer_step == total_steps:
                for parameter, average in zip(
                    model.parameters(), ema_parameters
                ):
                    parameter.copy_(average)
                for buffer, average in zip(
                    floating_buffers, ema_buffers
                ):
                    buffer.copy_(average)
>>>>>>> REPLACE