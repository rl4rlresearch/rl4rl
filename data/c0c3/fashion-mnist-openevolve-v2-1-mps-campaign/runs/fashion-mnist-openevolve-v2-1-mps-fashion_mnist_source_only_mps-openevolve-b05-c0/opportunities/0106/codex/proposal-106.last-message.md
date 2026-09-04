MECHANISM: Horizontal-flip test-time ensemble

HYPOTHESIS: Averaging logits for each validation image and its horizontal reflection will increase validation_correct above 9,258 by suppressing orientation-sensitive errors without changing training or learned parameters.

INTENDED_EDIT: Apply batched original-and-flipped inference during evaluation, average their logits, then retain the established 1.25 calibration multiplier.

EVIDENCE: Every completed logit-scaling experiment through 1.25 preserved exactly 9,258 predictions, showing that further scalar calibration can improve only the tie-breaker; flip ensembling can change argmax decisions while exploiting the approximate horizontal symmetry of the image classes.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            logits = logits * 1.25
        return logits
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.classifier(self.features(images))

        augmented = torch.cat(
            (images, torch.flip(images, dims=(-1,))), dim=0
        )
        logits = self.classifier(self.features(augmented))
        original_logits, flipped_logits = logits.chunk(2, dim=0)
        return 1.25 * (original_logits + flipped_logits) * 0.5
>>>>>>> REPLACE