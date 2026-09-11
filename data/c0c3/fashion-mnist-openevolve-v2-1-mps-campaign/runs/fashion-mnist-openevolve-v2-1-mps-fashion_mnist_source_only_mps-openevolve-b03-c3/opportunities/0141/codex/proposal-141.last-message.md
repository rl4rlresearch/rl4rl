MECHANISM: Zero-initialized global-evidence residual head

HYPOTHESIS: Adding a lightweight translation-stable global-average readout to the 9,284-correct mixed-pooling model will exceed 9,284 correct predictions by complementing the spatial classifier with shape-level evidence and a shorter optimization path.

INTENDED_EDIT: Add an 810-parameter linear head over globally averaged refined features, zero-initialize it to preserve the baseline’s initial logits, and add its logits at 0.25 strength to the existing classifier.

EVIDENCE: Scalar mixed pooling achieved the best result at 9,284 correct, while widening the shared classifier fell to 9,239 and squeeze-excitation fell to 9,263; this motivates adding narrowly targeted invariant supervision without altering the proven representation or expanding its main bottleneck.

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
        self.global_classifier = nn.Linear(80, 10)
        nn.init.zeros_(self.global_classifier.weight)
        nn.init.zeros_(self.global_classifier.bias)
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
        global_logits = self.global_classifier(features.mean(dim=(-2, -1)))
        features = torch.cat(
            (
                self.aggregate_local(features),
                self.aggregate_context(features),
            ),
            dim=1,
        )
        return self.classifier(features) + 0.25 * global_logits
>>>>>>> REPLACE