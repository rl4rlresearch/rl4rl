MECHANISM: Lossless phase-preserving convolutional pyramid with regional evidence logits

HYPOTHESIS: Replacing both max-pool reductions and the flatten-to-56 bottleneck with pixel-unshuffle downsampling, deep local residual mixing, and multiscale regional logits—while retaining the successful sparse 0.99 EMA—will exceed 9,334 correct predictions.

INTENDED_EDIT: Preserve every 2×2 sampling phase during downsampling, mix those phases with learned projections and seven dilated residual blocks, then predict directly from a 1×1/2×2/4×4 spatial pyramid.

EVIDENCE: Global pooling fell to 9,085 correct, showing spatial organization is load-bearing, while attention applied after the existing lossy pooling reached only 9,300. Reference Design 2’s sparse full-state EMA remains the best verified training protocol at 9,334 correct.

<<<<<<< SEARCH
class SpatialRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=False,
            padding_mode="replicate",
        )
        self.depthwise_norm = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
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
        mean_descriptor = F.adaptive_avg_pool2d(refined, output_size=1)
        peak_descriptor = F.adaptive_max_pool2d(refined, output_size=1)
        gate_features = 0.5 * (
            F.gelu(self.gate_down(mean_descriptor))
            + F.gelu(self.gate_down(peak_descriptor))
        )
        gate = 2.0 * torch.sigmoid(self.gate_up(gate_features))
        return F.gelu(images + gate * refined)
=======
class LocalResidual(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
            padding_mode="replicate",
        )
        self.depthwise_norm = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.pointwise_norm = nn.BatchNorm2d(channels)
        self.residual_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), 0.1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(images)
        residual = F.gelu(self.depthwise_norm(residual))
        residual = self.pointwise_norm(self.pointwise(residual))
        return F.gelu(images + self.residual_scale * residual)


class RegionalEvidenceHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(channels * (1 + 4 + 16), 10)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        regional_features = torch.cat(
            (
                F.adaptive_avg_pool2d(features, 1).flatten(1),
                F.adaptive_avg_pool2d(features, 2).flatten(1),
                F.adaptive_avg_pool2d(features, 4).flatten(1),
            ),
            dim=1,
        )
        regional_features = F.dropout(
            regional_features,
            p=0.10,
            training=self.training,
        )
        return self.classifier(regional_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            SpatialRefinement(64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 56),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(56, 10),
        )
=======
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.PixelUnshuffle(2),
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.PixelUnshuffle(2),
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            LocalResidual(128, dilation=1),
            LocalResidual(128, dilation=2),
            LocalResidual(128, dilation=1),
            LocalResidual(128, dilation=2),
            LocalResidual(128, dilation=1),
            LocalResidual(128, dilation=2),
            LocalResidual(128, dilation=1),
        )
        self.classifier = RegionalEvidenceHead(128)
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