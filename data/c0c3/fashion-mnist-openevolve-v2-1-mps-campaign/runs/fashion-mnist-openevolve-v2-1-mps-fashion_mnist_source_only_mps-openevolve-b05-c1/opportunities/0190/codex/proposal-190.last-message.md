MECHANISM: Learned channel-wise max–average downsampling

HYPOTHESIS: Replacing fixed max pooling with per-channel learned max–average mixtures will exceed 9,360 correct predictions by preserving distributed contour evidence while retaining peak responses where useful.

INTENDED_EDIT: Add two scale-specific adaptive pooling modules, initialized as exact max pooling, so training learns each channel’s preferred downsampling behavior with only 96 additional parameters.

EVIDENCE: Agreement calibration repeatedly improved cross-entropy without changing any of the 9,360 decisions, indicating that confidence scaling is no longer the primary limitation. The model currently assumes hard maxima are optimal for every feature channel; learning the downsampling rule changes the image representation directly and may reduce the translation sensitivity exposed by the extensive translated-view augmentation and inference ensemble.

<<<<<<< SEARCH
PEAK_LR = 3.3e-3


class ResidualBlock(nn.Module):
=======
PEAK_LR = 3.3e-3


class AdaptivePool2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.average_weight = nn.Parameter(torch.zeros(channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        maximum = F.max_pool2d(inputs, kernel_size=2)
        average = F.avg_pool2d(inputs, kernel_size=2)
        mixture = torch.tanh(
            self.average_weight
        )[None, :, None, None]
        return maximum + mixture * (average - maximum)


class ResidualBlock(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.GELU(),
            ResidualBlock(32, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            ResidualBlock(64, 64),
=======
            nn.GELU(),
            ResidualBlock(32, 32),
            AdaptivePool2d(32),
            ResidualBlock(32, 64),
            AdaptivePool2d(64),
            ResidualBlock(64, 64),
>>>>>>> REPLACE