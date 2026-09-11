MECHANISM: Late-training exponential weight consolidation

HYPOTHESIS: Applying a 0.99-decay EMA during the second half of training to Reference Design 3’s proven dual-statistic gated model will exceed 9,322 correct predictions by reducing late cosine-schedule parameter noise.

INTENDED_EDIT: Restore the 249,881-parameter mean/max refinement gate and calibrated temperature, then evaluate an EMA of parameters and floating-point normalization buffers accumulated over the second half of training.

EVIDENCE: Reference Design 3 achieved the best verified correctness at 9,322, while several further architectural and gating changes regressed; this retains that design exactly and tests an orthogonal optimization-level improvement without adding learned parameters.

<<<<<<< SEARCH
        self.pointwise_norm = nn.BatchNorm2d(channels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        refined = self.depthwise(images)
        refined = F.gelu(self.depthwise_norm(refined))
        refined = self.pointwise_norm(self.pointwise(refined))
        return F.gelu(images + refined)
=======
        self.pointwise_norm = nn.BatchNorm2d(channels)

        gate_channels = max(channels // 9, 1)
        self.gate_down = nn.Conv2d(
            channels,
            gate_channels,
            kernel_size=1,
        )
        self.gate_up = nn.Conv2d(
            gate_channels,
            channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.gate_up.weight)
        nn.init.zeros_(self.gate_up.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        refined = self.depthwise(images)
        refined = F.gelu(self.depthwise_norm(refined))
        refined = self.pointwise_norm(self.pointwise(refined))
        pooled = 0.5 * (
            F.adaptive_avg_pool2d(refined, output_size=1)
            + F.adaptive_max_pool2d(refined, output_size=1)
        )
        gate = 2.0 * torch.sigmoid(
            self.gate_up(F.gelu(self.gate_down(pooled)))
        )
        return F.gelu(images + gate * refined)
>>>>>>> REPLACE

<<<<<<< SEARCH
        ).log() / 0.753


def build_model() -> nn.Module:
=======
        ).log() / 0.75317


class EMAAdamW(torch.optim.AdamW):
    def __init__(self, model: nn.Module, total_steps: int) -> None:
        parameters = list(model.parameters())
        super().__init__(
            parameters,
            lr=PEAK_LR * 0.2,
            betas=(0.9, 0.99),
            weight_decay=5e-4,
        )
        self.tracked_tensors = parameters + [
            buffer
            for buffer in model.buffers()
            if buffer.is_floating_point()
        ]
        self.averaged_tensors = [
            tensor.detach().clone()
            for tensor in self.tracked_tensors
        ]
        self.total_steps = max(total_steps, 1)
        self.averaging_start = max(self.total_steps // 2, 1)
        self.update_count = 0
        self.ema_decay = 0.99

    def step(self, closure=None):
        loss = super().step(closure)
        self.update_count += 1
        with torch.no_grad():
            if self.update_count <= self.averaging_start:
                torch._foreach_copy_(
                    self.averaged_tensors,
                    self.tracked_tensors,
                )
            else:
                torch._foreach_lerp_(
                    self.averaged_tensors,
                    self.tracked_tensors,
                    1.0 - self.ema_decay,
                )
            if self.update_count >= self.total_steps:
                torch._foreach_copy_(
                    self.tracked_tensors,
                    self.averaged_tensors,
                )
        return loss


def build_model() -> nn.Module:
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=5e-4,
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    return EMAAdamW(model, total_steps)
>>>>>>> REPLACE