MECHANISM: Channel-adaptive mixed-statistic downsampling

HYPOTHESIS: Learning a separate max-versus-average pooling mixture for each feature channel will exceed 9,348 correct predictions by reducing translation-sensitive aliasing while retaining the spatial grid required by the classifier.

INTENDED_EDIT: Replace both hard max-pooling layers with channel-wise learned mixtures initialized near max pooling. This challenges the old assumption that one fixed downsampling statistic suits every feature channel; the new approach lets edge channels preserve peaks while broader shape channels learn smoother aggregation.

EVIDENCE: Translation TTA improved correctness from 9,344 to 9,348, indicating that one-pixel shift sensitivity remains load-bearing. Global pooling fell to 9,085, so this patch targets downsampling instability without discarding spatial organization. Its 96 additional learned parameters bring the verified 249,881-parameter model to 249,977, below the ceiling.

<<<<<<< SEARCH
        return F.gelu(images + gate * refined)


class ImageClassifier(nn.Module):
=======
        return F.gelu(images + gate * refined)


class ChannelMixedPool2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.max_weight_logits = nn.Parameter(
            torch.full((channels,), 2.0)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        maximum = F.max_pool2d(images, kernel_size=2, stride=2)
        average = F.avg_pool2d(images, kernel_size=2, stride=2)
        maximum_weight = torch.sigmoid(
            self.max_weight_logits
        ).view(1, -1, 1, 1)
        return average + maximum_weight * (maximum - average)


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
            ChannelMixedPool2d(32),
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
            ChannelMixedPool2d(64),
            SpatialRefinement(64),
>>>>>>> REPLACE