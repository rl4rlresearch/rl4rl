MECHANISM: Low-weight translation test-time ensemble

HYPOTHESIS: Blending 90% original logits with 10% mean logits from four one-pixel translations will preserve all 9,260 correct predictions while lowering validation cross-entropy below 0.212017.

INTENDED_EDIT: During evaluation, compute original and four replicate-padded translated views in one batched forward pass, blend their logits at 90%/10%, then apply the existing calibration multiplier.

EVIDENCE: The 50%-weighted translation ensemble improved cross-entropy to 0.210780 while losing only two correct predictions; reducing its influence to 10% should retain more of that calibration benefit with substantially less risk of crossing decision boundaries. The planned 20% test timed out without validation evidence.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            logits = logits * 1.4164
        return logits
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.classifier(self.features(images))

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        views = torch.cat(
            (
                images,
                padded[:, :, 1:29, 0:28],
                padded[:, :, 1:29, 2:30],
                padded[:, :, 0:28, 1:29],
                padded[:, :, 2:30, 1:29],
            ),
            dim=0,
        )
        view_logits = self.classifier(self.features(views))
        original, left, right, up, down = view_logits.chunk(5, dim=0)
        shifted = (left + right + up + down) * 0.25
        return (0.9 * original + 0.1 * shifted) * 1.4164
>>>>>>> REPLACE