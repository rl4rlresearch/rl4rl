MECHANISM: Parameter-neutral residual refinement

HYPOTHESIS: Adding skip connections around the existing same-width convolutions will improve limited-budget optimization and achieve at least 9,287 validation-correct predictions without increasing parameters or materially affecting runtime.

INTENDED_EDIT: Wrap the existing 32→32 and 64→64 convolutional refinements in residual connections while preserving all convolutional capacity, training settings, pooling, averaging, and TTA.

EVIDENCE: The 9,286-correct baseline remains strongest, terminal-pooling changes regressed, and an added residual refinement timed out; reusing existing convolutions as residual branches isolates the optimization benefit without its additional parameters or convolutional work.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
=======
class ResidualRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.gelu(features + self.norm(self.conv(features)))


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
            ResidualRefinement(32),
            nn.MaxPool2d(2),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
            ResidualRefinement(64),
            nn.MaxPool2d(2),
>>>>>>> REPLACE