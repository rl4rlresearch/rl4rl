MECHANISM: Flip-invariant test-time logit ensembling

HYPOTHESIS: Averaging predictions from each validation image and its horizontal reflection will exceed 9,290 correct predictions by reducing orientation-sensitive errors in the model already trained with 50% horizontal flips.

INTENDED_EDIT: Preserve training behavior while making evaluation average the original-view and flipped-view logits.

EVIDENCE: The current 237,818-parameter design reached 9,290 correct using horizontal-flip augmentation; evaluation currently uses only one orientation, so paired-view inference directly exploits the learned invariance without changing optimization or parameter count.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if self.training:
            return logits
        flipped_logits = self.classifier(self.features(images.flip(-1)))
        return 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE