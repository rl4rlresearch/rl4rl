MECHANISM: Mean-preserving stage-contrast mixed pooling

HYPOTHESIS: Coupling the two pooling gates around the verified 90%-max optimum while learning only their stage-to-stage difference will exceed 9,284 correct predictions by retaining useful stage specialization without noisy common-mode drift.

INTENDED_EDIT: Replace channelwise pooling coefficients with one shared contrast parameter that moves the two scalar stage gates in opposite directions around a fixed 90%-max center.

EVIDENCE: Independent 90%-max scalar gates achieved 9,284 correct, while one tied gate achieved 9,279 and 72 channelwise gates achieved 9,281; moving both scalar initializations to either 87.5% or 95% also regressed, motivating stage differentiation constrained around the established optimum.

<<<<<<< SEARCH
class ChannelwiseMixedPool2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.max_logits = nn.Parameter(
            torch.full((channels,), 2.1972246)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        max_weights = torch.sigmoid(self.max_logits)[None, :, None, None]
        maximum = F.max_pool2d(features, kernel_size=2, stride=2)
        average = F.avg_pool2d(features, kernel_size=2, stride=2)
        return max_weights * maximum + (1.0 - max_weights) * average
=======
class CenteredMixedPool2d(nn.Module):
    def __init__(
        self,
        stage_delta: nn.Parameter,
        direction: float,
    ) -> None:
        super().__init__()
        self.stage_delta = stage_delta
        self.direction = direction

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        max_weight = torch.sigmoid(
            2.1972246 + self.direction * self.stage_delta
        )
        maximum = F.max_pool2d(features, kernel_size=2, stride=2)
        average = F.avg_pool2d(features, kernel_size=2, stride=2)
        return max_weight * maximum + (1.0 - max_weight) * average
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            ChannelwiseMixedPool2d(24),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            ChannelwiseMixedPool2d(48),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
=======
        stage_delta = nn.Parameter(torch.zeros(()))
        self.features = nn.Sequential(
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            CenteredMixedPool2d(stage_delta, -1.0),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            CenteredMixedPool2d(stage_delta, 1.0),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE