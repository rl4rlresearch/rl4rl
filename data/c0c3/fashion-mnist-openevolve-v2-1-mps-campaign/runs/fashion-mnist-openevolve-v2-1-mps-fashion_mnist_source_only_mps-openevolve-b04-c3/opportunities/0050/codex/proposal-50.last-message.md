MECHANISM: Compact bilinear covariance shortcut

HYPOTHESIS: Adding a class-specific second-order channel-covariance branch to the verified 12.5%-floor model will exceed 9,240 correct predictions by capturing translation-tolerant feature co-occurrences that neither the narrow flattened head nor the failed first-order global-average shortcut represents.

INTENDED_EDIT: Preserve the proven flattened classifier, add a zero-initialized 22-channel bilinear covariance classifier for complementary logits, and restore the strongest verified 12.5% cosine floor; the model has 249,692 learned parameters.

EVIDENCE: The flattened model reached 9,240 correct, while its global-average shortcut fell to 9,230, challenging the shared assumption that first-order pooled features provide useful complementary context; compact covariance pooling instead retains 253 pairwise channel interactions with negligible spatial computation.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 38),
            nn.LayerNorm(38),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(38, 10),
        )

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.stem(images))
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 38),
            nn.LayerNorm(38),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(38, 10),
        )
        self.covariance_projection = nn.Conv2d(
            96, 22, kernel_size=1, bias=False
        )
        self.covariance_classifier = nn.Linear(22 * 23 // 2, 10)
        nn.init.zeros_(self.covariance_classifier.weight)
        nn.init.zeros_(self.covariance_classifier.bias)
        self.register_buffer(
            "covariance_indices",
            torch.triu_indices(22, 22),
            persistent=False,
        )

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        projected = F.gelu(self.covariance_projection(features)).flatten(2)
        projected = projected - projected.mean(dim=2, keepdim=True)
        covariance = torch.bmm(projected, projected.transpose(1, 2))
        covariance = covariance / projected.shape[2]
        covariance = covariance[
            :,
            self.covariance_indices[0],
            self.covariance_indices[1],
        ]
        covariance = torch.sign(covariance) * torch.sqrt(
            covariance.abs() + 1e-6
        )
        covariance = F.normalize(covariance, dim=1)
        return (
            self.classifier(features)
            + self.covariance_classifier(covariance)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    multiplier = 0.075 + 0.925 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.125 + 0.875 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE