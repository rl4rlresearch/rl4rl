MECHANISM: Late-training exponential checkpoint averaging

HYPOTHESIS: EMA-averaging the final 40% of the 9,284-correct mixed-pooling model’s trajectory will exceed 9,284 correct predictions by reducing late-step parameter and BatchNorm noise without changing the proven representation.

INTENDED_EDIT: Register an AdamW post-step hook that averages parameters and floating-point buffers with decay 0.98 from 60% training progress, then installs the averaged checkpoint after the final update.

EVIDENCE: Scalar mixed pooling is the best design at 9,284 correct, while added capacity, stronger consistency regularization, and smaller batches regressed or timed out; checkpoint averaging preserves that design and targets optimization variance instead.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
        betas=(0.9, 0.96),
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
        betas=(0.9, 0.96),
    )

    parameters = list(model.parameters())
    floating_buffers = [
        buffer for buffer in model.buffers() if buffer.is_floating_point()
    ]
    ema_start = max(1, int(0.60 * total_steps))
    ema_decay = 0.98
    ema_parameters = None
    ema_buffers = None
    completed_steps = 0

    @torch.no_grad()
    def update_ema(
        current_optimizer: torch.optim.Optimizer,
        args: tuple,
        kwargs: dict,
    ) -> None:
        del current_optimizer, args, kwargs
        nonlocal completed_steps, ema_parameters, ema_buffers
        completed_steps += 1

        if completed_steps == ema_start:
            ema_parameters = [
                parameter.detach().clone() for parameter in parameters
            ]
            ema_buffers = [
                buffer.detach().clone() for buffer in floating_buffers
            ]
        elif completed_steps > ema_start:
            assert ema_parameters is not None
            assert ema_buffers is not None
            for averaged, parameter in zip(ema_parameters, parameters):
                averaged.lerp_(parameter, 1.0 - ema_decay)
            for averaged, buffer in zip(ema_buffers, floating_buffers):
                averaged.lerp_(buffer, 1.0 - ema_decay)

        if completed_steps == total_steps and ema_parameters is not None:
            for parameter, averaged in zip(parameters, ema_parameters):
                parameter.copy_(averaged)
            for buffer, averaged in zip(floating_buffers, ema_buffers):
                buffer.copy_(averaged)

    optimizer.register_step_post_hook(update_ema)
    return optimizer
>>>>>>> REPLACE