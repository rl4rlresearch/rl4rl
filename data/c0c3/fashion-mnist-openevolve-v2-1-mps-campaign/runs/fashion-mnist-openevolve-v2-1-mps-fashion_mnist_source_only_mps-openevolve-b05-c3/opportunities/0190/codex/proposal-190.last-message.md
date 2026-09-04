MECHANISM: Identity-initialized global channel recalibration

HYPOTHESIS: Input-conditioned recalibration of the final 96 feature channels will exceed 9,210 correct predictions by using global image context to emphasize class-relevant texture and shape channels, while identity initialization and the qualified training/TTA pipeline preserve the strong baseline trajectory.

INTENDED_EDIT: Add a parameter-efficient squeeze/recalibration branch to the final residual stage, initialized as an exact identity, and restore the best-verified 1.178× output calibration; the model remains below the parameter ceiling at 249,002 parameters.

EVIDENCE: The qualified architecture plateaued at 9,210 correct despite extensive TTA tuning, while direct multi-scale feature fusion fell to 9,193; this suggests that merely exposing earlier features is unhelpful, motivating a distinct test of the shared assumption that final-stage channels should have fixed importance for every image.

<<<<<<< SEARCH
class ResidualStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.norm1(self.conv1(images)), inplace=True)
        refinement = self.norm2(self.conv2(features))
        return self.pool(F.silu(features + refinement, inplace=True))
=======
class ResidualStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channel_attention: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)

        if channel_attention:
            bottleneck_channels = max(8, out_channels // 6)
            self.attention_reduce = nn.Conv2d(
                out_channels,
                bottleneck_channels,
                kernel_size=1,
            )
            self.attention_expand = nn.Conv2d(
                bottleneck_channels,
                out_channels,
                kernel_size=1,
            )
            nn.init.zeros_(self.attention_expand.weight)
            nn.init.zeros_(self.attention_expand.bias)
        else:
            self.attention_reduce = None
            self.attention_expand = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.norm1(self.conv1(images)), inplace=True)
        refinement = self.norm2(self.conv2(features))
        features = F.silu(features + refinement, inplace=True)
        if self.attention_reduce is not None:
            context = F.adaptive_avg_pool2d(features, 1)
            context = F.silu(self.attention_reduce(context), inplace=True)
            modulation = torch.tanh(self.attention_expand(context))
            features = features * (1.0 + modulation)
        return self.pool(features)
>>>>>>> REPLACE

<<<<<<< SEARCH
            ResidualStage(64, 96),
=======
            ResidualStage(64, 96, channel_attention=True),
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.1875)
=======
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.178)
>>>>>>> REPLACE