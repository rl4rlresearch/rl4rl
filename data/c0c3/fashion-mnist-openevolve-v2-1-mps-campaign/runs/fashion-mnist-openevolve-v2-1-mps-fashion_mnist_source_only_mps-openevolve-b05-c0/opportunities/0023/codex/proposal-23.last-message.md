MECHANISM: Horizontal-reflection test-time ensembling

HYPOTHESIS: Averaging logits from each validation image and its horizontal reflection will exceed 9,254 correct predictions by reducing orientation-sensitive errors without altering the proven training trajectory.

INTENDED_EDIT: Preserve training behavior and add a two-view horizontal-flip ensemble only while the model is in evaluation mode.

EVIDENCE: The strided-EMA batch-32 design achieved 9,254 correct, while several attempts to alter training timed out; this validation-only change retains that successful procedure and adds negligible training cost.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if self.training:
            return logits
        flipped_logits = self.classifier(
            self.features(torch.flip(images, dims=(-1,)))
        )
        return 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE