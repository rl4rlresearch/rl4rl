MECHANISM: Joint invariant–disagreement view fusion

HYPOTHESIS: Jointly classifying the mirrored feature mean and absolute feature disagreement will exceed 9,286 correct predictions by learning when the two views corroborate or conflict, which independent-logit averaging cannot represent.

INTENDED_EDIT: Replace the assumption that mirrored views should be classified independently with an identity-initialized learned fusion of their invariant and disagreement features before the nonlinear classifier.

EVIDENCE: Mirrored ensembling raised accuracy substantially, while adding individual-view supervision regressed to 9,270 correct; this supports learning a genuinely joint representation rather than strengthening separate view predictions. The fusion adds only a 1×1 projection and reuses the existing two-view convolutional computation.

<<<<<<< SEARCH
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
=======
        self.pool = nn.MaxPool2d(2)
        self.view_fusion = nn.Conv2d(
            128, 64, kernel_size=1, bias=False
        )
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.classifier = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_view(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.residual1(features))
        features = self.pool(features)
        features = self.transition(features)
        features = F.gelu(features + self.residual2(features))
        features = self.pool(features)
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_view(images)
        if self.training:
            return logits
        flipped_logits = self._forward_view(torch.flip(images, dims=(-1,)))
        return 1.2112 * 0.5 * (logits + flipped_logits)
=======
    def _forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.residual1(features))
        features = self.pool(features)
        features = self.transition(features)
        features = F.gelu(features + self.residual2(features))
        return self.pool(features)

    def _classify_views(
        self,
        features: torch.Tensor,
        flipped_features: torch.Tensor,
    ) -> torch.Tensor:
        invariant = 0.5 * (features + flipped_features)
        disagreement = torch.abs(features - flipped_features)
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
        return self.classifier(fused)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_size = images.shape[0]
            paired_images = torch.cat(
                (images, torch.flip(images, dims=(-1,))),
                dim=0,
            )
            paired_features = self._forward_features(paired_images)
            features = paired_features[:batch_size]
            flipped_features = paired_features[batch_size:]
        else:
            features = self._forward_features(images)
            flipped_features = self._forward_features(
                torch.flip(images, dims=(-1,))
            )

        logits = self._classify_views(features, flipped_features)
        if self.training:
            return logits
        return 1.2112 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    batch_size = images.shape[0]
    paired_images = torch.cat(
        (images, torch.flip(images, dims=(-1,))),
        dim=0,
    )
    paired_logits = model(paired_images)
    logits = 0.5 * (
        paired_logits[:batch_size] + paired_logits[batch_size:]
    )
    return F.cross_entropy(logits, labels, label_smoothing=0.02)
=======
    del step, total_steps
    logits = model(images)
    return F.cross_entropy(logits, labels, label_smoothing=0.02)
>>>>>>> REPLACE