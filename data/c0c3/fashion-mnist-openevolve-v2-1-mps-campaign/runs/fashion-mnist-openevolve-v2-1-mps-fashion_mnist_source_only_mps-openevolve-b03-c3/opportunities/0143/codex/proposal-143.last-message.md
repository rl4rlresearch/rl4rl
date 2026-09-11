MECHANISM: Training-distribution-matched logit TTA

HYPOTHESIS: Restoring independent scalar 90%-max pooling and weighting validation translations exactly like the training augmentation distribution will exceed 9,284 correct predictions by removing an untrained directional and distance-dependent inference bias while preserving logit-space averaging.

INTENDED_EDIT: Replace the regressed channelwise pooling gates with the best verified stage-specific scalar gates, then use 0.09 for all radius-one cardinal shifts, 0.045 for diagonals, and 0.00625 for every radius-two shift.

EVIDENCE: Independent scalar mixed pooling achieved the best result at 9,284 correct, while channelwise pooling fell to 9,281 and probability-space TTA fell to 9,280; this motivates restoring the proven representation and testing only whether exact augmentation-matched logit weights improve inference.

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
        self.max_logit = nn.Parameter(torch.tensor(2.1972246))

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

<<<<<<< SEARCH
                if radius == 1:
                    if delta_x == 0:
                        weight = 0.10
                    elif delta_y == 0:
                        weight = 0.08
                    else:
                        weight = 0.045
                else:
                    manhattan_distance = abs(delta_y) + abs(delta_x)
                    if manhattan_distance == 2:
                        weight = 0.009375
                    elif manhattan_distance == 3:
                        weight = 0.00625
                    else:
                        weight = 0.003125
=======
                if radius == 1:
                    if delta_x == 0 or delta_y == 0:
                        weight = 0.09
                    else:
                        weight = 0.045
                else:
                    weight = 0.00625
>>>>>>> REPLACE