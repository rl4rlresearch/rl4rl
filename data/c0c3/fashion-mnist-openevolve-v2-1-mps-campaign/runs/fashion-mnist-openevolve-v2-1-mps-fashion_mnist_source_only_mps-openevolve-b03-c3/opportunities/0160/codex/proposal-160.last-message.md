MECHANISM: Contrast-conditioned channelwise mixed pooling

HYPOTHESIS: Replacing layer-wide pooling ratios with content-conditioned per-channel gates, initialized to reproduce the current 90% max-pooling behavior, will improve validation_score beyond 9284.413861719328 when combined with the best validated TTA weights.

INTENDED_EDIT: Give every channel its own baseline pooling logit and learned response to local peakiness, allowing edge and texture channels to downsample differently; restore Reference Design 1’s best-known center/vertical/horizontal TTA weighting. The model remains below the parameter ceiling at 245,294 learned parameters.

EVIDENCE: TTA refinements plateaued at 9,284 correct and increasingly small weight changes crossed a sharp boundary back to 9,283, indicating that fixed-view calibration is exhausted. All available designs assume one input-independent pooling ratio per layer; content-conditioned channelwise pooling instead changes the learned representation while starting exactly from the validated pooling behavior.

<<<<<<< SEARCH
class MixedPool2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_logit = nn.Parameter(torch.tensor(2.1972246))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        max_weight = torch.sigmoid(self.max_logit)
        maximum = F.max_pool2d(features, kernel_size=2, stride=2)
        average = F.avg_pool2d(features, kernel_size=2, stride=2)
        return max_weight * maximum + (1.0 - max_weight) * average
=======
class MixedPool2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.max_logit = nn.Parameter(
            torch.full((channels,), 2.1972246)
        )
        self.contrast_scale = nn.Parameter(torch.zeros(channels))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        maximum = F.max_pool2d(features, kernel_size=2, stride=2)
        average = F.avg_pool2d(features, kernel_size=2, stride=2)
        peakiness = maximum - average
        max_weight = torch.sigmoid(
            self.max_logit[None, :, None, None]
            + self.contrast_scale[None, :, None, None] * peakiness
        )
        return max_weight * maximum + (1.0 - max_weight) * average
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.GELU(),
            MixedPool2d(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            MixedPool2d(),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
=======
            nn.GELU(),
            MixedPool2d(24),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            MixedPool2d(48),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = 0.36 * self._flip_average(images)
=======
        logits = 0.36125 * self._flip_average(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
                if radius == 1:
                    if delta_x == 0:
                        weight = 0.11
                    elif delta_y == 0:
                        weight = 0.07
                    else:
                        weight = 0.045
=======
                if radius == 1:
                    if delta_x == 0:
                        weight = 0.10875
                    elif delta_y == 0:
                        weight = 0.070625
                    else:
                        weight = 0.045
>>>>>>> REPLACE