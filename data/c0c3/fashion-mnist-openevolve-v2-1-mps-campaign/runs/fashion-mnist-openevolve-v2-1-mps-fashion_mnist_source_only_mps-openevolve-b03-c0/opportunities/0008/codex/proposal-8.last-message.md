MECHANISM: Horizontal-reflection test-time logit ensemble

HYPOTHESIS: Averaging predictions from each validation image and its horizontal reflection will exceed 9,290 correct predictions by reducing orientation-sensitive errors while preserving the strongest verified training configuration.

INTENDED_EDIT: Keep training unchanged and, during evaluation only, average logits from the original and horizontally flipped images.

EVIDENCE: The 237,346-parameter model with batch size 64 is strongest at 9,290 correct; capacity expansion and EMA regressed, while the failed augmentation trial confounded reflection with two-pixel translations, motivating an isolated inference-time test of label-preserving reflection.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if self.training:
            return logits
        flipped_logits = self.classifier(
            self.features(torch.flip(images, dims=(3,)))
        )
        return 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE