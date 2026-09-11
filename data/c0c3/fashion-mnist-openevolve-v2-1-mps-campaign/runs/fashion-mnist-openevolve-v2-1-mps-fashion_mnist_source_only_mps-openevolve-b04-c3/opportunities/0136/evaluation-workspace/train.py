"""Editable image-classification research program.

The verification harness owns the data split, example budget, evaluation, and
timing.  This file owns the learned model and the trainable procedure.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0


class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.norm1(self.depthwise(inputs)))
        hidden = self.norm2(self.pointwise(hidden))
        return F.gelu(inputs + hidden)


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            ResidualDepthwiseBlock(48),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            ResidualDepthwiseBlock(96),
        )
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
        return self.classifier(self.stem(images))

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
            logits = 1.044 * logits
        return logits


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        weight_decay=1e-4,
    )
    optimizer._ema_model = model
    return optimizer


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        images.flip(-1),
        images,
    )
    return images, labels


def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels)


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = step / max(total_steps, 1)
    multiplier = 0.125 + 0.875 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
    ema_model = getattr(optimizer, "_ema_model", None)
    if ema_model is not None:
        ema_model.update_ema(decay=0.99)
