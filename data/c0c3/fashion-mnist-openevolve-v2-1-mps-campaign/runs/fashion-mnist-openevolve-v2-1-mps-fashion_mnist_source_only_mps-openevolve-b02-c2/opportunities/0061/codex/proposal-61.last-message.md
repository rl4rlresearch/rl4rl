MECHANISM: Evaluation-time exponential weight averaging

HYPOTHESIS: A ramped 0.99 parameter EMA on the faster 9,320-correct hard-maximum architecture will exceed 9,322 correct predictions by suppressing late-update noise while retaining enough runtime headroom to finish verification.

INTENDED_EDIT: Restore global-maximum channel saliency, maintain a low-overhead foreach EMA after optimizer steps, and swap the averaged parameters in for validation.

EVIDENCE: Hard-maximum attention achieved 9,320 correct in 75.3 seconds versus top-four attention’s 9,322 in 78.8 seconds; recent top-four and augmentation trials timed out, motivating an orthogonal stabilization method on the faster qualified architecture.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(48, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(48, 10),
        )

        self._ema_parameter_buffers: list[tuple[str, str]] = []
        for name, parameter in list(self.named_parameters()):
            buffer_name = "_ema_" + name.replace(".", "__")
            self.register_buffer(
                buffer_name,
                parameter.detach().clone(),
                persistent=False,
            )
            self._ema_parameter_buffers.append((name, buffer_name))
        self._ema_active = False
>>>>>>> REPLACE

<<<<<<< SEARCH
        channel_salient = features.flatten(2).topk(4, dim=2).values
        channel_salient = channel_salient.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
=======
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
=======
        return self.classifier(features)

    def _swap_ema_parameters(self) -> None:
        parameters = dict(self.named_parameters())
        with torch.no_grad():
            for name, buffer_name in self._ema_parameter_buffers:
                parameter = parameters[name]
                ema_parameter = getattr(self, buffer_name)
                temporary = parameter.detach().clone()
                parameter.copy_(ema_parameter)
                ema_parameter.copy_(temporary)

    def train(self, mode: bool = True) -> ImageClassifier:
        if mode and self._ema_active:
            self._swap_ema_parameters()
            self._ema_active = False
        elif not mode and not self._ema_active:
            self._swap_ema_parameters()
            self._ema_active = True
        return super().train(mode)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99),
    )
=======
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99),
    )
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
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    model = optimizer._ema_model
    parameters = dict(model.named_parameters())
    ema_parameters = [
        getattr(model, buffer_name)
        for _, buffer_name in model._ema_parameter_buffers
    ]
    current_parameters = [
        parameters[name]
        for name, _ in model._ema_parameter_buffers
    ]
    ema_decay = min(0.99, (1.0 + step) / (10.0 + step))
    with torch.no_grad():
        torch._foreach_lerp_(
            ema_parameters,
            current_parameters,
            1.0 - ema_decay,
        )

    progress = min(step / max(total_steps, 1), 1.0)
>>>>>>> REPLACE