MECHANISM: Parameter-neutral residual feature refinement

HYPOTHESIS: Adding identity shortcuts to the two same-width convolutional refinements will exceed 9,280 correct predictions by improving gradient flow and optimization within the fixed two-pass exposure, without materially increasing runtime or parameter count.

INTENDED_EDIT: Replace the 32-channel and 64-channel Conv-BatchNorm-GELU refinements with residual blocks containing the same learned layers.

EVIDENCE: Batch size 48 outperformed batch size 64 through additional optimizer updates, suggesting optimization is limiting; this change improves optimization while avoiding the extra convolutions that caused the larger residual pyramid to time out.

<<<<<<< SEARCH
GRAD_CLIP_NORM = 1.0


class ImageClassifier(nn.Module):
=======
GRAD_CLIP_NORM = 1.0


class ResidualRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(inputs)) + inputs)


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
=======
            ResidualRefinement(32),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
=======
            ResidualRefinement(64),
>>>>>>> REPLACE