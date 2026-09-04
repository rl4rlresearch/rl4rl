MECHANISM: Zero-initialized global-logit residual

HYPOTHESIS: Adding a direct global-average feature readout will exceed 9,257 correct predictions by supplying translation-robust garment evidence while preserving the proven coordinate-sensitive classifier.

INTENDED_EDIT: Add a bias-free, zero-initialized global-average classifier whose logits are summed with the existing flattened-head logits, increasing parameters from 233,434 to 233,994.

EVIDENCE: The 233,434-parameter widened convolutional model achieved the best result, while image-conditioned channel gating regressed to 9,204 and global self-attention timed out; a lightweight additive readout tests complementary global context without modulating the successful spatial representation or adding costly token mixing.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(56 * 7 * 7, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(56 * 7 * 7, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 10),
        )
        self.global_classifier = nn.Linear(56, 10, bias=False)
        nn.init.zeros_(self.global_classifier.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        return self.classifier(features)
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        spatial_logits = self.classifier(features)
        global_logits = self.global_classifier(features.mean(dim=(2, 3)))
        return spatial_logits + global_logits
>>>>>>> REPLACE