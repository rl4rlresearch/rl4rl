MECHANISM: EMA–endpoint probability ensemble with flip test-time augmentation

HYPOTHESIS: Restoring Reference Design 1 will increase validation_correct from 9,243 to 9,251 and reduce validation cross-entropy from 0.20883300514221192 to approximately 0.20655155601501465.

INTENDED_EDIT: Add 0.99 EMA tracking, blend EMA and live-model flip-ensembled probabilities at 75/25, and apply the verified 1.0495 inference scale.

EVIDENCE: Reference Design 1 verified this exact architecture and procedure at 9,251 correct and 0.20655155601501465 cross-entropy, the strongest completed result provided.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 38),
            nn.LayerNorm(38),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(38, 10),
        )

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 38),
            nn.LayerNorm(38),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(38, 10),
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
>>>>>>> REPLACE

<<<<<<< SEARCH
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
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._classify(images)
        if not self.training:
            flipped_logits = self._classify(images.flip(-1))
            power = 0.5831695556640625
            ema_logits = (
                torch.logaddexp(
                    power * F.log_softmax(logits, dim=1),
                    power * F.log_softmax(flipped_logits, dim=1),
                )
                - math.log(2.0)
            ) / power

            if self._using_ema:
                self._swap_ema_state()
                try:
                    live_logits = self._classify(images)
                    live_flipped_logits = self._classify(images.flip(-1))
                    live_logits = (
                        torch.logaddexp(
                            power * F.log_softmax(live_logits, dim=1),
                            power * F.log_softmax(live_flipped_logits, dim=1),
                        )
                        - math.log(2.0)
                    ) / power
                finally:
                    self._swap_ema_state()

                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.75),
                    F.log_softmax(live_logits, dim=1) + math.log(0.25),
                )
            else:
                logits = ema_logits
            logits = 1.0495 * logits
        return logits
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