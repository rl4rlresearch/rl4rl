MECHANISM: Annealed single-view auxiliary supervision

HYPOTHESIS: Adding low-weight per-view supervision during early training and annealing it away during EMA collection will exceed 9,328 correct predictions by strengthening the successful invariant feature pathway while preserving late specialization of the fused classifier.

INTENDED_EDIT: Reuse the shared fusion and classifier heads to classify each mirrored view independently, add their cross-entropy at weight 0.15, and linearly remove that auxiliary loss over the second half of training.

EVIDENCE: Replacing the full fusion with disagreement-only residual fusion fell to 9,290 correct, showing that the transformed invariant pathway is essential; dense-head widening also fell to 9,300, motivating parameter-neutral supervision rather than additional capacity.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 10),
        )
        self.auxiliary_logits = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(fused)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
=======
        return self.classifier(fused)

    def _classify_individual_views(
        self,
        features: torch.Tensor,
        flipped_features: torch.Tensor,
    ) -> torch.Tensor:
        individual_features = torch.cat(
            (features, flipped_features),
            dim=0,
        )
        fused = self.view_fusion(
            torch.cat(
                (
                    individual_features,
                    torch.zeros_like(individual_features),
                ),
                dim=1,
            )
        )
        return self.classifier(fused)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self._classify_views(features, flipped_features)
        if self.training:
            return logits
        return 1.2112 * logits
=======
        logits = self._classify_views(features, flipped_features)
        if self.training:
            self.auxiliary_logits = self._classify_individual_views(
                features,
                flipped_features,
            )
            return logits
        self.auxiliary_logits = None
        return 1.2112 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - dropout_decay)
    return F.cross_entropy(
        logits,
        labels,
        label_smoothing=label_smoothing,
    )
=======
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - dropout_decay)
    main_loss = F.cross_entropy(
        logits,
        labels,
        label_smoothing=label_smoothing,
    )
    auxiliary_loss = F.cross_entropy(
        model.auxiliary_logits,
        labels.repeat(2),
        label_smoothing=label_smoothing,
    )
    auxiliary_weight = 0.15 * (1.0 - dropout_decay)
    return main_loss + auxiliary_weight * auxiliary_loss
>>>>>>> REPLACE