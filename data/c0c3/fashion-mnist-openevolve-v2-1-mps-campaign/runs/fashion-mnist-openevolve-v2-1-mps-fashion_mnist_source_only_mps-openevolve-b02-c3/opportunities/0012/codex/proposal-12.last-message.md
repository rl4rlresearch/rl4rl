MECHANISM: Parameter-efficient residual spatial refinement

HYPOTHESIS: A depthwise-separable residual block at 7×7 resolution, while preserving the 56-unit location-sensitive head and using matched axial augmentation, will exceed 9,247 correct predictions.

INTENDED_EDIT: Add a 4,928-parameter late residual refinement block, retain the wider dense head, and adopt the proven center-weight-three axial training and inference distribution.

EVIDENCE: A full late 64-channel convolution improved accuracy from 9,245 to 9,247 despite forcing the dense head from 56 to 46 units; residual depthwise-separable refinement tests the same useful late processing while retaining spatial-head capacity and staying below 250,000 parameters.

<<<<<<< SEARCH
WARMUP_FRACTION = 0.08


class ImageClassifier(nn.Module):
=======
WARMUP_FRACTION = 0.08


class SpatialRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.depthwise_norm = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.pointwise_norm = nn.BatchNorm2d(channels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        refined = self.depthwise(images)
        refined = F.gelu(self.depthwise_norm(refined))
        refined = self.pointwise_norm(self.pointwise(refined))
        return F.gelu(images + refined)


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
=======
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            SpatialRefinement(64),
        )
        self.classifier = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
        crop_weights = (2.75, 1.0, 1.0, 1.0, 1.0)
=======
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    padding = 2
    padded = F.pad(
        images,
        (padding, padding, padding, padding),
        mode="replicate",
    )
    offsets_y = torch.randint(
        0, 2 * padding + 1, (batch, 1), device=images.device
    )
    offsets_x = torch.randint(
        0, 2 * padding + 1, (batch, 1), device=images.device
    )
=======
    padding = 1
    padded = F.pad(
        images,
        (padding, padding, padding, padding),
        mode="replicate",
    )
    translation = torch.randint(0, 7, (batch,), device=images.device)
    offsets_y = torch.tensor(
        (1, 1, 1, 0, 2, 1, 1), device=images.device
    )[translation].unsqueeze(1)
    offsets_x = torch.tensor(
        (1, 1, 1, 1, 1, 0, 2), device=images.device
    )[translation].unsqueeze(1)
>>>>>>> REPLACE