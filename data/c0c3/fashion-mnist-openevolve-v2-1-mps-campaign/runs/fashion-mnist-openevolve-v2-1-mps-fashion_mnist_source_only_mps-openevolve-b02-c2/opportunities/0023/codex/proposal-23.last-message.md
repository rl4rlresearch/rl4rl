MECHANISM: Periodic exponential weight averaging on the top-four attention model

HYPOTHESIS: Evaluating a 0.90-decay EMA sampled every ten optimizer steps will exceed 9,322 correct predictions by suppressing late-training parameter noise while preserving the best verified architecture and training objective.

INTENDED_EDIT: Restore shared average-plus-top-four channel attention, maintain a parameter-neutral EMA of floating model state during training, and use that averaged state for flip-ensemble validation.

EVIDENCE: Top-four channel attention produced the best verified result at 9,322 correct; descriptor reparameterization, translation, and added spatial refinement regressed, motivating a conservative optimization-level refinement that leaves the successful computation and paired cross-entropy intact.

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

        ema_sources = [
            ("parameter", name, value)
            for name, value in self.named_parameters()
        ]
        ema_sources.extend(
            ("buffer", name, value)
            for name, value in self.named_buffers()
            if value.is_floating_point()
        )
        self._ema_entries: list[tuple[str, str, str]] = []
        for index, (kind, name, value) in enumerate(ema_sources):
            ema_name = f"_ema_state_{index}"
            self.register_buffer(
                ema_name,
                value.detach().clone(),
                persistent=False,
            )
            self._ema_entries.append((kind, name, ema_name))
        self._ema_ready = False
        self._ema_applied = False
>>>>>>> REPLACE

<<<<<<< SEARCH
        channel_summary = F.adaptive_avg_pool2d(features, 1)
        channel_summary = channel_summary.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_summary)
=======
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_salient = features.flatten(2).topk(4, dim=2).values
        channel_salient = channel_salient.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not self.training and self._ema_ready and not self._ema_applied:
            with torch.no_grad():
                for kind, name, ema_name in self._ema_entries:
                    if kind == "parameter":
                        destination = self.get_parameter(name)
                    else:
                        destination = self.get_buffer(name)
                    destination.copy_(self.get_buffer(ema_name))
            self._ema_applied = True

        logits = self._forward_once(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99),
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
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
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
) -> None:
    model = optimizer._ema_model
    if step % 10 == 0:
        with torch.no_grad():
            for kind, name, ema_name in model._ema_entries:
                if kind == "parameter":
                    source = model.get_parameter(name)
                else:
                    source = model.get_buffer(name)
                model.get_buffer(ema_name).lerp_(source.detach(), 0.10)
        model._ema_ready = True
        model._ema_applied = False

    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
>>>>>>> REPLACE