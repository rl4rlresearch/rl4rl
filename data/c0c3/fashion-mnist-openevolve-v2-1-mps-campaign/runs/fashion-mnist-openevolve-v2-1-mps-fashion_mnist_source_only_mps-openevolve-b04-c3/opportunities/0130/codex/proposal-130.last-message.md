MECHANISM: Coarse-grid mean/max evidence pooling with a wider nonlinear head

HYPOTHESIS: Replacing precise 7×7 flattening with joint 3×3 mean/max descriptors and widening the head from 38 to 105 units, while retaining the qualified EMA–endpoint ensemble, will exceed 9,251 correct predictions or tie it with lower cross-entropy.

INTENDED_EDIT: Replace the assumption that most capacity should encode exact final-grid coordinates with translation-tolerant coarse spatial evidence. The unchanged stem feeds adaptive mean/max pooling, a 105-wide classifier, and the verified 0.99 EMA plus 25% live-model probability blend. The model has 248,599 learned parameters.

EVIDENCE: Calibration-only changes plateaued at 9,243 correct, whereas EMA–endpoint blending reached 9,251. The earlier horizontal-pooling design timed out after also adding residual computation; this patch keeps the convolutional workload unchanged and uses approximately the same dense multiply count as the current head, isolating the alternative representation.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 38),
            nn.LayerNorm(38),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(38, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 2 * 3 * 3, 105),
            nn.LayerNorm(105),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(105, 10),
        )

        self._ema_state_names: list[str] = []
        for index, (name, value) in enumerate(self.state_dict().items()):
            self.register_buffer(
                f"_ema_state_{index}",
                value.detach().clone(),
                persistent=False,
            )
            self._ema_state_names.append(name)
        self._ema_initialized = False
        self._using_ema = False
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.stem(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._classify(images)
        if not self.training:
            flipped_logits = self._classify(images.flip(-1))
            power = 0.5831695556640625
            logits = 1.03592 * (
                torch.logaddexp(
                    power * F.log_softmax(logits, dim=1),
                    power * F.log_softmax(flipped_logits, dim=1),
                )
                - math.log(2.0)
            ) / power
        return logits
=======
    @torch.no_grad()
    def update_ema(self, decay: float) -> None:
        live_state = self.state_dict()
        shadows = [
            getattr(self, f"_ema_state_{index}")
            for index in range(len(self._ema_state_names))
        ]
        sources = [live_state[name] for name in self._ema_state_names]

        if not self._ema_initialized:
            for shadow, source in zip(shadows, sources):
                shadow.copy_(source)
            self._ema_initialized = True
            return

        float_shadows = []
        float_sources = []
        for shadow, source in zip(shadows, sources):
            if shadow.is_floating_point():
                float_shadows.append(shadow)
                float_sources.append(source)
            else:
                shadow.copy_(source)
        torch._foreach_lerp_(float_shadows, float_sources, 1.0 - decay)

    @torch.no_grad()
    def _swap_ema_state(self) -> None:
        live_state = self.state_dict()
        for index, name in enumerate(self._ema_state_names):
            live_value = live_state[name]
            shadow = getattr(self, f"_ema_state_{index}")
            temporary = live_value.detach().clone()
            live_value.copy_(shadow)
            shadow.copy_(temporary)
        self._using_ema = not self._using_ema

    def train(self, mode: bool = True) -> ImageClassifier:
        if mode and self._using_ema:
            self._swap_ema_state()
        result = super().train(mode)
        if not mode and self._ema_initialized and not self._using_ema:
            self._swap_ema_state()
        return result

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        pooled = torch.cat(
            (
                F.adaptive_avg_pool2d(features, output_size=(3, 3)),
                F.adaptive_max_pool2d(features, output_size=(3, 3)),
            ),
            dim=1,
        )
        return self.classifier(pooled)

    def _flip_ensemble(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._classify(images)
        flipped_logits = self._classify(images.flip(-1))
        power = 0.5831695556640625
        return (
            torch.logaddexp(
                power * F.log_softmax(logits, dim=1),
                power * F.log_softmax(flipped_logits, dim=1),
            )
            - math.log(2.0)
        ) / power

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._classify(images)

        ema_logits = self._flip_ensemble(images)
        if self._using_ema:
            self._swap_ema_state()
            try:
                live_logits = self._flip_ensemble(images)
            finally:
                self._swap_ema_state()
            logits = torch.logaddexp(
                F.log_softmax(ema_logits, dim=1) + math.log(0.75),
                F.log_softmax(live_logits, dim=1) + math.log(0.25),
            )
        else:
            logits = ema_logits
        return 1.03592 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        weight_decay=1e-4,
    )
    optimizer._ema_model = model
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
    ema_model = getattr(optimizer, "_ema_model", None)
    if ema_model is not None:
        ema_model.update_ema(decay=0.99)
>>>>>>> REPLACE