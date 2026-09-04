MECHANISM: Global first/second-order feature-statistics decoder

HYPOTHESIS: Replacing fixed-coordinate flattening with translation-invariant channel means and covariances will exceed 9,247 correct predictions by capturing global shape and texture relationships while better matching the crop/flip ensemble.

INTENDED_EDIT: Preserve the proven convolutional extractor and parameter count, but transform its 48×7×7 output into 48 channel means plus a 48×48 covariance matrix before the existing classifier.

EVIDENCE: The phase-preserving space-to-depth stem regressed to 9,191 and increasing flattened-head capacity regressed to 9,210, challenging the load-bearing assumption that more spatially indexed detail improves classification; second-order pooling instead models learned feature co-occurrences without adding capacity.

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        return self.classifier(features)
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        spatial_features = features.flatten(2)
        feature_mean = spatial_features.mean(dim=2)
        centered_features = spatial_features - feature_mean.unsqueeze(2)
        covariance = torch.bmm(
            centered_features,
            centered_features.transpose(1, 2),
        ) / spatial_features.shape[2]
        statistics = torch.cat(
            (feature_mean, covariance.flatten(1)),
            dim=1,
        )
        statistics = torch.sign(statistics) * torch.sqrt(
            torch.abs(statistics) + 1.0e-5
        )
        return self.classifier(statistics)
>>>>>>> REPLACE