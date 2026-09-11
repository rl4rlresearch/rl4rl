MECHANISM: Zero-initialized invariant residual classification branch

HYPOTHESIS: Adding a direct global mean–max decision path to the proven spatial classifier will exceed 9,280 correct predictions by capturing translation-tolerant feature presence without the runtime cost of the timed-out residual pyramid.

INTENDED_EDIT: Preserve the successful feature extractor and position-specific head, but augment every prediction with a learned, zero-initialized linear residual computed from global mean and maximum channel statistics.

EVIDENCE: Repeated optimizer and capacity changes failed or timed out, while the global mean–max design was never accuracy-tested because its seven-convolution extractor exceeded the time limit; this patch isolates that alternative prediction mechanism with only 1,920 parameters and negligible training computation.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 10),
        )
        self.global_classifier = nn.Linear(96 * 2, 10, bias=False)
        nn.init.zeros_(self.global_classifier.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if self.training:
=======
    def _predict(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        local_logits = self.classifier(features)
        global_features = torch.cat(
            (features.mean(dim=(2, 3)), features.amax(dim=(2, 3))), dim=1
        )
        return local_logits + self.global_classifier(global_features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._predict(images)
        if self.training:
>>>>>>> REPLACE

<<<<<<< SEARCH
            view_logits = self.classifier(self.features(view))
=======
            view_logits = self._predict(view)
>>>>>>> REPLACE

<<<<<<< SEARCH
            flipped_logits = self.classifier(self.features(view.flip(-1)))
=======
            flipped_logits = self._predict(view.flip(-1))
>>>>>>> REPLACE