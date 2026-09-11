MECHANISM: Single-convolution residual feature refinement

HYPOTHESIS: Adding an identity shortcut around the existing 48-channel convolution will exceed 9,254 correct predictions by improving finite-budget optimization without increasing parameter count or computational cost materially.

INTENDED_EDIT: Wrap the third convolution and BatchNorm in a residual block, preserving the proven optimizer, EMA, loss, batch size, evaluation scaling, and 239,634-parameter budget.

EVIDENCE: Accuracy improved monotonically as optimizer-step count increased from batch 128 to 64 to 32, indicating optimization within the fixed exposure is limiting; a parameter-neutral residual path directly improves gradient and feature propagation.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
=======
class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.gelu(inputs + self.norm(self.conv(inputs)))


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            ResidualConvBlock(48),
            nn.MaxPool2d(2),
>>>>>>> REPLACE