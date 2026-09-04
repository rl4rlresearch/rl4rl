MECHANISM: Conservative one-pixel translation test-time ensemble

HYPOTHESIS: Averaging the original evaluation logits with lower-weight logits from four one-pixel translations will increase validation_correct above 9,258 by reducing pooling-phase sensitivity while preserving orientation-dependent class information.

INTENDED_EDIT: During evaluation only, combine the original prediction at double weight with predictions from replicated-border shifts in all four axial directions, then apply the established 1.25 calibration.

EVIDENCE: Horizontal-reflection ensembling reduced validation_correct to 8,883, showing orientation must be preserved; the one-pixel training-translation attempt timed out and supplied no contradictory accuracy evidence, while translation averaging can change argmax predictions without increasing training time or parameters.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            logits = logits * 1.25
        return logits
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
            shifted_images = (
                padded[:, :, 1:29, 0:28],
                padded[:, :, 1:29, 2:30],
                padded[:, :, 0:28, 1:29],
                padded[:, :, 2:30, 1:29],
            )
            shifted_logits = sum(
                self.classifier(self.features(shifted))
                for shifted in shifted_images
            )
            logits = ((2.0 * logits + shifted_logits) / 6.0) * 1.25
        return logits
>>>>>>> REPLACE