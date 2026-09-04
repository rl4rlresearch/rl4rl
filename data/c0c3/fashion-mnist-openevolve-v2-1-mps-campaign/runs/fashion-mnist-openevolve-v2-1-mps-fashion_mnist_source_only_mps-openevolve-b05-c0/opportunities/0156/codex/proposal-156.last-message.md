MECHANISM: Evaluation-time horizontal-reflection logit ensemble

HYPOTHESIS: Averaging original-view and horizontally reflected-view logits will exceed 9,260 correct predictions by reducing orientation-sensitive errors without perturbing the verified training trajectory.

INTENDED_EDIT: During evaluation only, average logits from each image and its horizontal reflection before applying the verified 1.4164 calibration multiplier.

EVIDENCE: Training-time augmentation reduced correctness, while optimization, loss, capacity, and EMA changes either regressed or timed out; an evaluation-only ensemble tests invariance without changing learned weights or training dynamics.

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
            reflected = torch.flip(images, dims=(-1,))
            reflected_logits = self.classifier(self.features(reflected))
            logits = 0.5 * (logits + reflected_logits)
            logits = logits * 1.4164
        return logits
>>>>>>> REPLACE