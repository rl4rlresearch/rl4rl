MECHANISM: Compact second-order channel-covariance readout

HYPOTHESIS: Adding a lightweight global covariance classifier will exceed 9,284 correct predictions by capturing class-specific feature co-occurrences that the existing first-order spatial readout cannot express directly.

INTENDED_EDIT: Preserve the validated convolutional and spatial-classifier path while adding a 16-channel signed-square-root covariance head, learned residual logit mixing, and the best verified evaluation calibration; total learned parameters become 249,035.

EVIDENCE: Pointwise refinement fell to 9,230 correct and channel-specific pooling fell to 9,282, while full spatial attention exceeded the time limit; this challenges the shared assumption that independently transformed local activations are sufficient using a computationally compact global pairwise representation.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 140),
            nn.LayerNorm(140),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(140, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 140),
            nn.LayerNorm(140),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(140, 10),
        )
        self.second_order = nn.Sequential(
            nn.Conv2d(80, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
        self.second_order_classifier = nn.Linear(16 * 16, 10)
        self.second_order_logit = nn.Parameter(torch.tensor(-1.0986123))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        features = torch.cat(
            (
                self.aggregate_local(features),
                self.aggregate_context(features),
            ),
            dim=1,
        )
        return self.classifier(features)
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))

        second_order = self.second_order(features).flatten(2)
        second_order = second_order - second_order.mean(dim=2, keepdim=True)
        covariance = torch.bmm(
            second_order, second_order.transpose(1, 2)
        ) / max(second_order.size(2) - 1, 1)
        covariance = torch.sign(covariance) * torch.sqrt(
            covariance.abs() + 1.0e-6
        )
        covariance = F.normalize(covariance.flatten(1), dim=1)
        second_order_logits = self.second_order_classifier(covariance)

        features = torch.cat(
            (
                self.aggregate_local(features),
                self.aggregate_context(features),
            ),
            dim=1,
        )
        spatial_logits = self.classifier(features)
        second_order_weight = torch.sigmoid(self.second_order_logit)
        return spatial_logits + second_order_weight * second_order_logits
>>>>>>> REPLACE

<<<<<<< SEARCH
        return 1.1645 * logits
=======
        return 1.16727 * logits
>>>>>>> REPLACE