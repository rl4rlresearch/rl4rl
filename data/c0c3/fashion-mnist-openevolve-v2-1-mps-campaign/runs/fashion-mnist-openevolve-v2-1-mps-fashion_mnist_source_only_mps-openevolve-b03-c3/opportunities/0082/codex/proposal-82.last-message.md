MECHANISM: Tail-trajectory exponential weight averaging

HYPOTHESIS: Applying a 0.98 EMA over the final 35% of optimization will exceed 9,268 correct predictions by reducing endpoint noise while retaining the verified 40-local/24-context representation.

INTENDED_EDIT: Preserve the best architecture, restore its mildly distance-stratified radius-2 TTA, and evaluate an EMA of parameters and BatchNorm state accumulated during the low-learning-rate training tail.

EVIDENCE: The 40/24 model produced the best 9,268-correct result, while capacity, fusion, padding, and augmentation changes lost accuracy; the mild TTA redistribution preserved all 9,268 predictions and improved cross-entropy, motivating an optimization-stability change on that baseline.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 140),
            nn.LayerNorm(140),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(140, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 140),
            nn.LayerNorm(140),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(140, 10),
        )
        self._ema_state = None
        self._ema_applied = False
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        height, width = images.shape[-2:]
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._ema_applied = False
            return self._forward_once(images)

        if self._ema_state is not None and not self._ema_applied:
            with torch.no_grad():
                for name, current in self.state_dict().items():
                    average = self._ema_state.get(name)
                    if (
                        average is not None
                        and average.device == current.device
                        and average.dtype == current.dtype
                    ):
                        current.copy_(average)
            self._ema_applied = True

        height, width = images.shape[-2:]
>>>>>>> REPLACE

<<<<<<< SEARCH
                else:
                    weight = 0.00625
                logits = logits + weight * self._flip_average(shifted)
=======
                else:
                    manhattan_distance = abs(delta_y) + abs(delta_x)
                    if manhattan_distance == 2:
                        weight = 0.009375
                    elif manhattan_distance == 3:
                        weight = 0.00625
                    else:
                        weight = 0.003125
                logits = logits + weight * self._flip_average(shifted)
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    del total_steps
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    del total_steps
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
    )
    model._ema_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    optimizer._ema_model = model
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    model = optimizer._ema_model
    start_step = int(0.65 * total_steps)
    with torch.no_grad():
        for name, current in model.state_dict().items():
            average = model._ema_state.get(name)
            if (
                average is None
                or average.device != current.device
                or average.dtype != current.dtype
            ):
                model._ema_state[name] = current.detach().clone()
            elif step < start_step or not current.is_floating_point():
                average.copy_(current)
            else:
                average.lerp_(current, 0.02)

    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
>>>>>>> REPLACE