MECHANISM: Hybrid spatial–covariance classification head

HYPOTHESIS: Combining fixed-position shape evidence with centered channel-covariance evidence will exceed 9,348 correct predictions by distinguishing classes with similar silhouettes but different feature co-occurrence patterns.

INTENDED_EDIT: Replace the single 56-unit flattened bottleneck with a 43-unit spatial branch plus a zero-initialized second-order covariance branch, retaining full 7×7 layout and fitting 249,940 learned parameters.

EVIDENCE: Global pooling fell to 9,085 correct, establishing that layout must remain available; axial TTA reached 9,348 by improving alignment, but subsequent fusion and calibration changes did not improve correctness. This challenges the shared assumption that class predictions need only first-order features at fixed positions while preserving the proven spatial pathway.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
=======
class SpatialCovarianceHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 7 * 7, 43),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(43, 10),
        )
        self.covariance_classifier = nn.Linear(
            channels * channels,
            10,
        )
        nn.init.zeros_(self.covariance_classifier.weight)
        nn.init.zeros_(self.covariance_classifier.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spatial_logits = self.spatial_classifier(features)

        activations = features.flatten(2)
        activations = activations - activations.mean(
            dim=2,
            keepdim=True,
        )
        covariance = torch.bmm(
            activations,
            activations.transpose(1, 2),
        ) / activations.shape[2]
        covariance = torch.sign(covariance) * torch.sqrt(
            covariance.abs() + 1.0e-5
        )
        covariance = 64.0 * F.normalize(
            covariance.flatten(1),
            dim=1,
        )
        return spatial_logits + self.covariance_classifier(covariance)


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
        self.classifier = SpatialCovarianceHead(64)
>>>>>>> REPLACE