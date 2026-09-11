MECHANISM: One-pixel translation test-time ensembling

HYPOTHESIS: Averaging logits across the original image and four one-pixel translations will exceed 9,279 correct predictions by reducing position-sensitive errors while preserving class semantics better than horizontal reflection.

INTENDED_EDIT: During evaluation only, average logits from the original and four cardinal one-pixel translations, then apply the best verified 1.17817 calibration.

EVIDENCE: Calibration and batch-normalization searches plateaued at 9,279 correct, while horizontal-reflection ensembling fell to 8,384 because reflection is not reliably label preserving; small translations provide a distinct, semantics-preserving prediction-changing probe.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            logits = 1.178165 * logits
        return logits
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            translated_images = (
                F.pad(images, (1, 0, 0, 0))[:, :, :, :28],
                F.pad(images, (0, 1, 0, 0))[:, :, :, 1:],
                F.pad(images, (0, 0, 1, 0))[:, :, :28, :],
                F.pad(images, (0, 0, 0, 1))[:, :, 1:, :],
            )
            translated_logits = [
                self.classifier(self.features(translated))
                for translated in translated_images
            ]
            logits = torch.stack([logits, *translated_logits]).mean(dim=0)
            logits = 1.17817 * logits
        return logits
>>>>>>> REPLACE