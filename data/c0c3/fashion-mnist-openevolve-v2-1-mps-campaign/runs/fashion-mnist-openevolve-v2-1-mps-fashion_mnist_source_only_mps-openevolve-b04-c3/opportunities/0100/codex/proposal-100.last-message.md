MECHANISM: Anisotropically pooled full-channel nonlinear head

HYPOTHESIS: Preserving all 96 semantic channels while pooling the redundant 7×7 grid to a vertically detailed 4×2 layout and widening the learned head from 38 to 235 units will exceed 9,243 correct predictions within the parameter ceiling.

INTENDED_EDIT: Replace the full-resolution narrow bottleneck with adaptive 4×2 spatial pooling and a 235-unit nonlinear classifier, using 249,329 learned parameters; also apply the best verified argmax-preserving logit scale.

EVIDENCE: The current 9,243-correct design spends most parameters mapping 4,704 spatial-channel values into only 38 units. The 9,235-correct alternative instead compressed channels from 96 to 24 while retaining every location; this patch tests the complementary assumption that late semantic channels are more valuable than fine spatial resolution.

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
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 2, 235),
            nn.LayerNorm(235),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(235, 10),
        )

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        return self.classifier(self.spatial_pool(features))
>>>>>>> REPLACE

<<<<<<< SEARCH
            logits = (
                torch.logaddexp(
=======
            logits = 1.03592 * (
                torch.logaddexp(
>>>>>>> REPLACE