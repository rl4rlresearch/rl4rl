MECHANISM: Equivariance-cached cyclic translation augmentation

HYPOTHESIS: Efficiently applying translations of up to two pixels during the first 75% of training will exceed 9,328 correct predictions by improving positional robustness while retaining a clean late fine-tuning phase.

INTENDED_EDIT: Replace redundant mirrored convolution passes with exact feature-map flipping, and replace redundant horizontal-flip augmentation with deterministic batchwise replicate-padded translations that add minimal runtime overhead.

EVIDENCE: Per-image translation augmentation timed out without accuracy evidence, while equivariance-cached feature extraction completed in 62.99 seconds; this tests the same promising invariance mechanism using a cheaper translation implementation.

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

<<<<<<< SEARCH
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        torch.flip(images, dims=(-1,)),
        images,
    )
    return images, labels
=======
) -> tuple[torch.Tensor, torch.Tensor]:
    if step < 3 * total_steps // 4:
        shift_x = step % 5 - 2
        shift_y = (step // 5) % 5 - 2
        padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
        images = padded[
            ...,
            2 + shift_y : 30 + shift_y,
            2 + shift_x : 30 + shift_x,
        ]
    return images, labels
>>>>>>> REPLACE