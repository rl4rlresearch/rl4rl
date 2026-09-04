MECHANISM: Contrast-adaptive local downsampling

HYPOTHESIS: Replacing fixed max pooling with channel-learned, locally contrast-adaptive max/average pooling will exceed 9,348 correct predictions by reducing one-pixel phase sensitivity while retaining sharp discriminative responses and spatial organization.

INTENDED_EDIT: Add a parameter-efficient adaptive pooling operator at both downsampling stages and restore the best-verified 1.3125/0.6875 vertical-favoring TTA weights.

EVIDENCE: Vertical crop reweighting improved correctness from 9,344 to 9,348, indicating load-bearing sensitivity to one-pixel alignment; meanwhile global pooling fell to 9,085 and the hypercolumn reached only 9,309, so the alternative preserves the full spatial grid and changes only how local features are downsampled.

<<<<<<< SEARCH
        return F.gelu(images + gate * refined)


class ImageClassifier(nn.Module):
=======
        return F.gelu(images + gate * refined)


class ContrastAdaptivePool2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.contrast_scale = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        maximum = F.max_pool2d(images, kernel_size=2, stride=2)
        average = F.avg_pool2d(images, kernel_size=2, stride=2)
        contrast = maximum - average
        gate = torch.sigmoid(
            math.log(3.0) + self.contrast_scale * contrast
        )
        return average + gate * contrast


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
=======
            nn.BatchNorm2d(32),
            nn.GELU(),
            ContrastAdaptivePool2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            SpatialRefinement(64),
=======
            nn.BatchNorm2d(64),
            nn.GELU(),
            ContrastAdaptivePool2d(64),
            SpatialRefinement(64),
>>>>>>> REPLACE

<<<<<<< SEARCH
        crop_weights = (3.0, 1.25, 1.25, 0.75, 0.75)
=======
        crop_weights = (3.0, 1.3125, 1.3125, 0.6875, 0.6875)
>>>>>>> REPLACE