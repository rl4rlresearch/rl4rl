MECHANISM: Center-weighted vertical translation ensemble

HYPOTHESIS: Averaging the original prediction with half-weight contributions from one-pixel upward and downward views will exceed 9,328 correct predictions by reducing residual vertical-alignment sensitivity without altering training.

INTENDED_EDIT: Preserve training exactly, but during evaluation ensemble logits from the original image and symmetric one-pixel vertical translations, weighting the original view 0.50 and each translation 0.25.

EVIDENCE: Training-time translation augmentation repeatedly exceeded the runtime limit, while numerous runtime-neutral changes to EMA, label smoothing, dropout, capacity, and dilation failed to improve 9,328 correct; evaluation-only translation tests the still-unresolved geometric-tolerance hypothesis on only 10,000 examples with no backward pass.

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
        if self.training:
            return logits
        return 1.2112 * logits
=======
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
            return self._classify_views(features, flipped_features)

        batch_size = images.shape[0]
        shifted_up = torch.cat(
            (images[..., 1:, :], images[..., -1:, :]),
            dim=-2,
        )
        shifted_down = torch.cat(
            (images[..., :1, :], images[..., :-1, :]),
            dim=-2,
        )
        view_images = torch.cat(
            (images, shifted_up, shifted_down),
            dim=0,
        )
        features = self._forward_features(view_images)
        flipped_features = self._forward_features(
            torch.flip(view_images, dims=(-1,))
        )
        view_logits = self._classify_views(
            features,
            flipped_features,
        ).reshape(3, batch_size, 10)
        logits = (
            0.50 * view_logits[0]
            + 0.25 * view_logits[1]
            + 0.25 * view_logits[2]
        )
        return 1.2112 * logits
>>>>>>> REPLACE