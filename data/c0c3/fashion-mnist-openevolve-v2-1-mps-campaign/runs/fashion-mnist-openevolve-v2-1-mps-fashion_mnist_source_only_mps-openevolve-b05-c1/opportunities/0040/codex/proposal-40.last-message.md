MECHANISM: Class-specific factorized spatial evidence

HYPOTHESIS: Replacing the shared 30-feature flattened bottleneck with deeper spatial processing and four fixed-position evidence components per class will exceed 9,249 correct predictions by preserving spatial geometry while learning class-specific channel–position templates.

INTENDED_EDIT: Add a residual block at 7×7 resolution and replace the dense head with a rank-four, class-specific spatial evidence head; training and calibrated test-time augmentation remain unchanged.

EVIDENCE: Content-addressed pooling fell to 9,228 correct, showing that input-dependent spatial aggregation was harmful, while widening the fixed-position bottleneck fell to 9,213, showing that more late dense capacity was insufficient. This tests a different mechanism that retains fixed spatial structure but moves representation learning into the convolutional trunk and computes each class from its own distributed evidence templates.

<<<<<<< SEARCH
            ResidualBlock(64, 64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 30),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(30, 10),
        )
=======
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )
        self.num_evidence_components = 4
        self.evidence_maps = nn.Conv2d(
            64,
            10 * self.num_evidence_components,
            kernel_size=1,
        )
        self.spatial_logits = nn.Parameter(
            torch.zeros(10, self.num_evidence_components, 7, 7)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        evidence = self.evidence_maps(features).reshape(
            images.shape[0], 10, self.num_evidence_components, 7, 7
        )
        spatial_weights = F.softmax(
            self.spatial_logits.flatten(-2), dim=-1
        ).reshape(1, 10, self.num_evidence_components, 7, 7)
        component_logits = (evidence * spatial_weights).sum(dim=(-2, -1))
        return component_logits.mean(dim=-1)
>>>>>>> REPLACE