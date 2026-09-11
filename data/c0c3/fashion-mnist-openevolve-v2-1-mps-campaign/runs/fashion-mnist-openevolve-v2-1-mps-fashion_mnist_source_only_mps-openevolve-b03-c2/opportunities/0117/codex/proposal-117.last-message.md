MECHANISM: Warm-started exponential weight averaging

HYPOTHESIS: Evaluating a warm-started EMA of the unchanged training trajectory will exceed 9,256 correct predictions by reducing late-step optimizer variance without altering the established augmentation, loss, or inference ensemble.

INTENDED_EDIT: Maintain an EMA shadow of every learned parameter after each AdamW step and temporarily use those averaged parameters during evaluation.

EVIDENCE: Changes to ensemble-aligned loss, stronger smoothing, and longer shift augmentation reduced validation_correct, while the current recipe reached 9,256; this motivates preserving that exact trajectory and averaging its late checkpoints instead of perturbing its objective or data distribution.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(128, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(128, 10),
        )

        parameters = tuple(self.parameters())
        self._ema_parameter_count = len(parameters)
        self._ema_updates = 0
        for index, parameter in enumerate(parameters):
            self.register_buffer(
                f"_ema_parameter_{index}",
                parameter.detach().clone(),
                persistent=False,
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        log_prob_views = []
        for view_index, view in enumerate(views):
            paired = torch.cat((view, view.flip(-1)), dim=0)
            original, mirrored = F.log_softmax(
                self._forward_once(paired), dim=1
            ).chunk(2, dim=0)
            log_weight = math.log(3.25 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.5)
        return 1.35605 * aggregate_logits
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def _ema_parameters(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            getattr(self, f"_ema_parameter_{index}")
            for index in range(self._ema_parameter_count)
        )

    @torch.no_grad()
    def update_ema(self) -> None:
        self._ema_updates += 1
        decay = min(
            0.995,
            (1.0 + self._ema_updates) / (10.0 + self._ema_updates),
        )
        for averaged, parameter in zip(
            self._ema_parameters(), self.parameters()
        ):
            averaged.lerp_(parameter.detach(), 1.0 - decay)

    def _forward_ensemble(self, images: torch.Tensor) -> torch.Tensor:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        log_prob_views = []
        for view_index, view in enumerate(views):
            paired = torch.cat((view, view.flip(-1)), dim=0)
            original, mirrored = F.log_softmax(
                self._forward_once(paired), dim=1
            ).chunk(2, dim=0)
            log_weight = math.log(3.25 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.5)
        return 1.35605 * aggregate_logits

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        parameters = tuple(self.parameters())
        live_data = tuple(parameter.data for parameter in parameters)
        try:
            for parameter, averaged in zip(
                parameters, self._ema_parameters()
            ):
                parameter.data = averaged.data
            return self._forward_ensemble(images)
        finally:
            for parameter, data in zip(parameters, live_data):
                parameter.data = data
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        betas=(0.9, 0.99),
        weight_decay=3.0e-4,
    )
=======
class EMAAdamW(torch.optim.AdamW):
    def __init__(self, model: ImageClassifier, **kwargs) -> None:
        super().__init__(model.parameters(), **kwargs)
        self._ema_model = model

    def step(self, closure=None):
        loss = super().step(closure)
        self._ema_model.update_ema()
        return loss


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return EMAAdamW(
        model,
        lr=2.5e-3,
        betas=(0.9, 0.99),
        weight_decay=3.0e-4,
    )
>>>>>>> REPLACE