MECHANISM: Sparse late-training exponential weight consolidation

HYPOTHESIS: Applying an eight-step approximation of 0.99-decay EMA to Reference Design 3 during the second half of training will exceed 9,322 correct predictions without repeating the per-step EMA experiment’s timeout.

INTENDED_EDIT: Restore the proven equal mean/max refinement gate and calibrated temperature, then sparsely average parameters and floating-point normalization buffers and install the averaged state after the final optimizer step.

EVIDENCE: Reference Design 3 achieved the best verified correctness at 9,322, while later architectural and loss changes regressed. Per-step late EMA was promising but timed out, motivating an eightfold-lower-frequency approximation of that orthogonal optimization idea.

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
=======
        ).log() / 0.75317
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
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=5e-4,
    )
    optimizer._averaging_model = model
    optimizer._averaged_state = {}
    optimizer._averaging_last_step = None
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = PEAK_LR * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = PEAK_LR * multiplier

    should_average = progress >= 0.5 and (
        step % 8 == 0 or step >= total_steps
    )
    if should_average:
        model = optimizer._averaging_model
        averaged_state = optimizer._averaged_state
        last_step = optimizer._averaging_last_step
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if not torch.is_floating_point(value):
                    continue
                if name not in averaged_state:
                    averaged_state[name] = value.detach().clone()
                else:
                    elapsed = max(step - last_step, 1)
                    decay = 0.99 ** elapsed
                    averaged_state[name].mul_(decay).add_(
                        value.detach(),
                        alpha=1.0 - decay,
                    )
        optimizer._averaging_last_step = step

        if step >= total_steps:
            with torch.no_grad():
                for name, value in model.state_dict().items():
                    if name in averaged_state:
                        value.copy_(averaged_state[name])
>>>>>>> REPLACE