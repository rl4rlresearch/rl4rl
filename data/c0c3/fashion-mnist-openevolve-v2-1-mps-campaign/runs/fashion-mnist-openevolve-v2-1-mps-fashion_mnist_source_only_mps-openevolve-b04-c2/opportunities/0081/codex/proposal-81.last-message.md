MECHANISM: Residual pairwise feature refinement

HYPOTHESIS: Adding an identity skip around the existing grouped 5×5 refinement will improve gradient flow and preserve useful third-stage features, raising validation_correct above 9,279 within the fixed two-pass exposure budget.

INTENDED_EDIT: Wrap the final grouped convolution, batch normalization, and GELU in a parameter-neutral residual module; leave optimization, calibration, and example accounting unchanged.

EVIDENCE: Calibration has plateaued at 9,279 correct, while translation TTA and augmentation reduced accuracy, motivating a new prediction-changing architectural probe that retains the best-known training procedure.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
=======
class ResidualPairwiseRefinement(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            128, 128, kernel_size=5, padding=2, groups=64, bias=False
        )
        self.norm = nn.BatchNorm2d(128, momentum=0.00623359375)
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        refined = self.norm(self.conv(features))
        return self.activation(features + refined)


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.00623359375),
            nn.GELU(),
=======
            ResidualPairwiseRefinement(),
>>>>>>> REPLACE