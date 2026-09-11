MECHANISM: Residual global-context channel recalibration

HYPOTHESIS: Identity-initialized channel recalibration will exceed 9,287 correct predictions by letting global image context modulate semantic feature channels without the harmful spatial selection seen in prior pooling experiments.

INTENDED_EDIT: Add a bottleneck channel-recalibration block before global pooling, reduce the classifier width to keep 249,997 parameters, and retain the best verified view-agreement calibration.

EVIDENCE: Spatial attention fell to 9,266 correct and static spatial pooling to 9,252, suggesting location selection is harmful; channel recalibration instead preserves all spatial evidence while introducing learned context-dependent feature interactions.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
=======
class ChannelRecalibration(nn.Module):
    def __init__(self, channels: int, bottleneck: int) -> None:
        super().__init__()
        self.reduce = nn.Linear(channels, bottleneck)
        self.expand = nn.Linear(bottleneck, channels)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        context = F.adaptive_avg_pool2d(features, 1).flatten(1)
        gates = 2.0 * torch.sigmoid(
            self.expand(F.gelu(self.reduce(context)))
        )
        return features * gates[:, :, None, None]


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.late_context = MultiScaleBlock(96)
        self.late_refinement = SpatialRefinement(96)
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 61),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(61, 10),
        )
=======
        self.late_context = MultiScaleBlock(96)
        self.late_refinement = SpatialRefinement(96)
        self.channel_recalibration = ChannelRecalibration(96, 6)
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 55),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(55, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = self.late_context(features)
        features = self.late_refinement(features)
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
=======
        features = self.late_context(features)
        features = self.late_refinement(features)
        features = self.channel_recalibration(features)
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pooled_logits = (
            1.557939852073332 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.557939852073332
        return 1.22775 * pooled_logits
=======
        pooled_logits = (
            1.5578022218542173 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5578022218542173
        pooled_predictions = pooled_logits.argmax(dim=-1)
        view_agreement = (
            view_logits.argmax(dim=-1) == pooled_predictions[None, None, :]
        ).float().mean(dim=(0, 1))
        calibration = 1.22775 * (
            0.92211476 + 0.07788524 * view_agreement
        )
        return calibration.unsqueeze(1) * pooled_logits
>>>>>>> REPLACE