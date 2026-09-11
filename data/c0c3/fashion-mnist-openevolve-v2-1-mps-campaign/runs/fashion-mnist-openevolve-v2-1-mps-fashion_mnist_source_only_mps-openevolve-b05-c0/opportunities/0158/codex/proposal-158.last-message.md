MECHANISM: Low-weight translation test-time ensemble

HYPOTHESIS: Weighting the verified four-shift ensemble at only 20% will preserve or exceed the baseline’s 9,260 correct predictions while retaining some of the cross-entropy improvement observed with the overly strong 50% ensemble.

INTENDED_EDIT: During evaluation, blend 80% original-view logits with 20% mean logits from four replicate-padded one-pixel translations, then apply the existing calibration multiplier.

EVIDENCE: The 50%-weighted translation ensemble reached 9,258 correct—only two below baseline—while improving cross-entropy from 0.212017 to 0.210780, motivating a smaller interpolation that perturbs fewer decisions.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            logits = logits * 1.4164
        return logits
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
            translated = torch.cat(
                (
                    padded[:, :, 1:29, 0:28],
                    padded[:, :, 1:29, 2:30],
                    padded[:, :, 0:28, 1:29],
                    padded[:, :, 2:30, 1:29],
                ),
                dim=0,
            )
            translated_logits = self.classifier(self.features(translated))
            translated_logits = translated_logits.reshape(
                4, images.shape[0], 10
            ).mean(dim=0)
            logits = (0.8 * logits + 0.2 * translated_logits) * 1.4164
        return logits
>>>>>>> REPLACE