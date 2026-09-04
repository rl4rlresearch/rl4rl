MECHANISM: Disagreement-gated margin-adaptive flip fusion

HYPOTHESIS: The verified 0.0275 adjustment will retain 9,331 correct predictions while reducing validation cross-entropy from 0.1962515151977539 to approximately 0.1962239959716797.

INTENDED_EDIT: Replace fixed evaluation-time flip weighting with the best qualified adaptive fusion rule, shifting weight toward the higher-margin view only during decisive view disagreements.

EVIDENCE: Reference Design 2 completed in 69.25 seconds with 9,331 correct and 0.1962239959716797 cross-entropy, the best qualified validation_score reported.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.184 * (
                0.4914921875 * logits + 0.5085078125 * flipped_logits
            )
        return logits
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))

            original_top2 = logits.topk(2, dim=1)
            flipped_top2 = flipped_logits.topk(2, dim=1)
            original_margin = (
                original_top2.values[:, 0] - original_top2.values[:, 1]
            )
            flipped_margin = (
                flipped_top2.values[:, 0] - flipped_top2.values[:, 1]
            )
            margin_difference = flipped_margin - original_margin

            disagreement = (
                original_top2.indices[:, 0] != flipped_top2.indices[:, 0]
            )
            decisive = disagreement & (margin_difference.abs() > 0.5)
            adjustment = (
                0.0275
                * torch.tanh(0.5 * margin_difference)
                * decisive.to(logits.dtype)
            )
            flipped_weight = 0.5085009765625 + adjustment
            logits = 1.184 * (
                (1.0 - flipped_weight[:, None]) * logits
                + flipped_weight[:, None] * flipped_logits
            )
        return logits
>>>>>>> REPLACE