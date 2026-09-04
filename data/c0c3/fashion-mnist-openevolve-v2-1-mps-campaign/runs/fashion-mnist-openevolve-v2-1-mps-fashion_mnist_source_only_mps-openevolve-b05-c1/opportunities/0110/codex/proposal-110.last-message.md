MECHANISM: Channel-adaptive mixed pooling

HYPOTHESIS: Replacing hard max-only downsampling with learned per-channel max/average mixtures will exceed 9,360 correct predictions by reducing shift aliasing while retaining salient local responses.

INTENDED_EDIT: Add a lightweight mixed-pooling layer and use it for both spatial reductions, adding only 96 learned parameters.

EVIDENCE: The best design relies heavily on translated training views and multi-shift evaluation, suggesting that hard downsampling remains a load-bearing source of shift sensitivity. This tests invariance inside the learned hierarchy rather than revisiting unverified head aggregation changes.

<<<<<<< SEARCH
PEAK_LR = 3.3e-3


class ResidualBlock(nn.Module):
=======
PEAK_LR = 3.3e-3


class ChannelMixedPool(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.max_mix_logits = nn.Parameter(torch.ones(channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        maximum = F.max_pool2d(inputs, kernel_size=2, stride=2)
        average = F.avg_pool2d(inputs, kernel_size=2, stride=2)
        maximum_weight = torch.sigmoid(
            self.max_mix_logits
        )[None, :, None, None]
        return average + maximum_weight * (maximum - average)


class ResidualBlock(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            ResidualBlock(32, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            ResidualBlock(64, 64),
        )
=======
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            ResidualBlock(32, 32),
            ChannelMixedPool(32),
            ResidualBlock(32, 64),
            ChannelMixedPool(64),
            ResidualBlock(64, 64),
        )
>>>>>>> REPLACE