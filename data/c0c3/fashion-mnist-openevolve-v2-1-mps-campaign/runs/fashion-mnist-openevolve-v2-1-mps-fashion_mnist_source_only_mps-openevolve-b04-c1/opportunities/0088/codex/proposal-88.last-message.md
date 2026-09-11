MECHANISM: Equivariance-cached late logit pooling

HYPOTHESIS: Shared-head classification of each mirrored orientation before logit averaging will exceed 9,328 correct predictions by retaining asymmetric spatial evidence, while deriving the second feature map through exact horizontal equivariance removes the runtime bottleneck that prevented the prior late-pooling design from finishing.

INTENDED_EDIT: Remove early invariant/disagreement fusion, classify the original and horizontally reflected feature maps independently with one shared head, average their logits, and compute the convolutional features only once.

EVIDENCE: The prior late-logit-pooling attempt directly tested this more expressive invariant representation but timed out; the convolutional feature extractor consists entirely of horizontally equivariant operations, so reflecting its output avoids the redundant second convolutional pass that caused repeated mirrored-view designs to approach the time limit.

<<<<<<< SEARCH
        self.pool = nn.MaxPool2d(2)
        self.view_fusion = nn.Conv2d(
            128, 64, kernel_size=1, bias=False
        )
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.classifier = nn.Sequential(
=======
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    def _classify_views(
        self,
        features: torch.Tensor,
        flipped_features: torch.Tensor,
    ) -> torch.Tensor:
        return 0.5 * (
            self.classifier(features)
            + self.classifier(flipped_features)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self._forward_features(images)
        flipped_features = torch.flip(features, dims=(-1,))

        logits = self._classify_views(features, flipped_features)
>>>>>>> REPLACE