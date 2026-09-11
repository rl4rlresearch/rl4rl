MECHANISM: Residual class-specific spatial logit shortcut

HYPOTHESIS: Reallocating the shared 38-unit bottleneck into a 28-unit nonlinear branch plus a direct full-resolution class-specific branch will exceed 9,243 correct predictions while remaining within the existing runtime and parameter envelope.

INTENDED_EDIT: Preserve the verified convolutional stem and training procedure, but compute logits by summing nonlinear latent predictions with learned class-specific templates over every channel and 7×7 location; use the best verified flip-ensemble order.

EVIDENCE: The current model reaches 9,243 correct while concentrating 179,256 weights in a shared spatial bottleneck, whereas the 9,166-correct attention design indicates that discarding spatial detail is harmful. Unlike the timed-out spatial-refinement head, this reallocation preserves all locations without adding convolutional work and totals 244,920 parameters.

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
        self.nonlinear_classifier = nn.Sequential(
            nn.Linear(96 * 7 * 7, 28),
            nn.LayerNorm(28),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(28, 10),
        )
        self.spatial_classifier = nn.Linear(96 * 7 * 7, 10)

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images).flatten(1)
        return (
            self.nonlinear_classifier(features)
            + self.spatial_classifier(features)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            power = 0.625
=======
            power = 0.5831695556640625
>>>>>>> REPLACE