MECHANISM: Identity-centered squeeze-and-excitation

HYPOTHESIS: Adding parameter-efficient channel attention while preserving all 64 convolutional channels will exceed 9,290 correct predictions by learning image-dependent feature emphasis within the remaining parameter budget.

INTENDED_EDIT: Add a 2,128-parameter squeeze-and-excitation module after the final pooling layer, bringing the model to 249,964 parameters without changing training or evaluation.

EVIDENCE: Expanding the dense bottleneck to 58 units produced the 9,290-correct best result, while reallocating convolutional width to a larger head fell to 9,264; this motivates using the remaining budget for adaptive capacity without sacrificing the validated convolutional width.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
=======
class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, bottleneck: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(channels, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        scale = self.gate(images.mean(dim=(2, 3)))
        return images * (2.0 * scale[:, :, None, None])


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
=======
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
            SqueezeExcitation(64, 16),
        )
>>>>>>> REPLACE