MECHANISM: Latent-subtype log-mean-exp classifier

HYPOTHESIS: Replacing each class’s single affine output with four learned subtype templates will exceed 9,311 correct predictions by modeling multimodal within-class appearances while remaining under the parameter ceiling.

INTENDED_EDIT: Narrow the embedding from 147 to 142 units and replace the ten-output linear layer with 40 component logits, aggregating four components per class through log-mean-exp during training and TTA.

EVIDENCE: Temperature calibration plateaued at exactly 9,311 correct and probability-level TTA reduced correctness, so progress requires different learned decision boundaries. The current design’s load-bearing assumption is one affine template per class; this patch tests smooth multi-region class scores without the costly residual, attention, bilinear, or multiscale mechanisms that failed to verify.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 147),
            nn.BatchNorm1d(147),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(147, 10),
        )
=======
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 142),
            nn.BatchNorm1d(142),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.components = nn.Linear(142, 10 * 4)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if self.training:
            return logits

        padded = F.pad(images, (1, 1, 1, 1))
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        ensemble = logits * 2.0
        for view in views[1:]:
            view_logits = self.classifier(self.features(view))
            ensemble = ensemble + view_logits
        for view_index, view in enumerate(views):
            flipped_logits = self.classifier(self.features(view.flip(-1)))
            if view_index == 0:
                flipped_logits = flipped_logits * 2.0
            ensemble = ensemble + flipped_logits
        return ensemble / (12.0 * 0.9350)
=======
    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding(self.features(images))
        component_logits = self.components(embedding).view(-1, 10, 4)
        return torch.logsumexp(component_logits, dim=-1) - math.log(4.0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._classify(images)
        if self.training:
            return logits

        padded = F.pad(images, (1, 1, 1, 1))
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        ensemble = logits * 2.0
        for view in views[1:]:
            ensemble = ensemble + self._classify(view)
        for view_index, view in enumerate(views):
            flipped_logits = self._classify(view.flip(-1))
            if view_index == 0:
                flipped_logits = flipped_logits * 2.0
            ensemble = ensemble + flipped_logits
        return ensemble / (12.0 * 0.9350)
>>>>>>> REPLACE