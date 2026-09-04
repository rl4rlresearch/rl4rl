MECHANISM: Stage-specific 95%-max learnable mixed pooling

HYPOTHESIS: Initializing independent stage-level mixtures halfway between the 9,284-correct 90%-max design and hard max pooling will exceed 9,284 correct by retaining anti-aliasing while preserving more edge-selective evidence during early optimization.

INTENDED_EDIT: Replace channelwise pooling coefficients with one independent coefficient per stage and initialize each mixture to 95% max pooling; preserve all other settings.

EVIDENCE: Stage-specific 90%-max pooling achieved the best result at 9,284 correct, while channelwise coefficients fell to 9,281 and a shared coefficient fell to 9,279, identifying stage-level granularity as best and motivating a local interpolation toward hard max.

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
class MixedPool2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_logit = nn.Parameter(torch.tensor(2.944439))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        max_weight = torch.sigmoid(self.max_logit)
        maximum = F.max_pool2d(features, kernel_size=2, stride=2)
        average = F.avg_pool2d(features, kernel_size=2, stride=2)
        return max_weight * maximum + (1.0 - max_weight) * average
>>>>>>> REPLACE

<<<<<<< SEARCH
            ChannelwiseMixedPool2d(24),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            ChannelwiseMixedPool2d(48),
=======
            MixedPool2d(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            MixedPool2d(),
>>>>>>> REPLACE