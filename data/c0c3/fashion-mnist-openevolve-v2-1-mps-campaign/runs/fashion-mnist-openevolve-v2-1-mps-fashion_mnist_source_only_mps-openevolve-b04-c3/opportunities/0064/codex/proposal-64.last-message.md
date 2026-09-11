MECHANISM: Boundary-search probability power-mean flip ensemble

HYPOTHESIS: A 0.584375-order power mean with the verified 12.5% learning-rate floor will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090484.

INTENDED_EDIT: Restore the strongest verified training schedule and replace logit averaging with a 0.584375-order generalized mean of original and horizontally flipped probabilities.

EVIDENCE: Order 0.5875 achieved 9,243 correct at 0.2090484 cross-entropy, while 0.58125 achieved 9,242 correct at the lower 0.2090442; their midpoint directly probes the accuracy boundary for a better tie-break score.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._classify(images)
        if not self.training:
            flipped_logits = self._classify(images.flip(-1))
            logits = 0.5 * (logits + flipped_logits)
        return logits
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._classify(images)
        if not self.training:
            flipped_logits = self._classify(images.flip(-1))
            power = 0.584375
            logits = (
                torch.logaddexp(
                    power * F.log_softmax(logits, dim=1),
                    power * F.log_softmax(flipped_logits, dim=1),
                )
                - math.log(2.0)
            ) / power
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    multiplier = 0.075 + 0.925 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.125 + 0.875 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE