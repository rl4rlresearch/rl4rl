MECHANISM: Layout-preserving spatial-pyramid classification head

HYPOTHESIS: Replacing the narrow absolute-position flattening head with a wider multiscale regional representation will exceed 9,348 correct predictions by reducing translation sensitivity without discarding image layout.

INTENDED_EDIT: Pool the 7×7 feature map into 1×1, 2×2, and 4×4 regional descriptors, concatenate them, and classify through a 130-unit head; this keeps the model below 250,000 parameters with similar classifier compute.

EVIDENCE: Translation TTA improved correctness from 9,344 to 9,348, indicating sensitivity to exact spatial alignment, while global pooling fell to 9,085 because it erased layout. The old assumption is that all 49 absolute feature positions should feed a narrow 56-unit bottleneck; the new approach preserves a 4×4 layout while adding coarse regional context and a wider learned readout.

<<<<<<< SEARCH
        return F.gelu(images + gate * refined)


class ImageClassifier(nn.Module):
=======
        return F.gelu(images + gate * refined)


class SpatialPyramidClassifier(nn.Module):
    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        pyramid_features = channels * (1 + 4 + 16)
        self.classifier = nn.Sequential(
            nn.Linear(pyramid_features, hidden),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(hidden, 10),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pyramid = torch.cat(
            tuple(
                F.adaptive_avg_pool2d(features, output_size=size).flatten(1)
                for size in (1, 2, 4)
            ),
            dim=1,
        )
        return self.classifier(pyramid)


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 56),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(56, 10),
        )
=======
        self.classifier = SpatialPyramidClassifier(
            channels=64,
            hidden=130,
        )
>>>>>>> REPLACE