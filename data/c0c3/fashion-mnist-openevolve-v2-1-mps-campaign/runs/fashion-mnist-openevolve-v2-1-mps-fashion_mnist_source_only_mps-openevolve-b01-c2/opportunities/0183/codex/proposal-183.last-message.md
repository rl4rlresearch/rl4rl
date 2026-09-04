MECHANISM: Verified high-disparity margin-adaptive flip fusion

HYPOTHESIS: The qualified 0.025 disagreement-gated adjustment will retain 9,331 correct predictions while reducing validation cross-entropy from 0.19625150680541992 to approximately 0.1962260711669922.

INTENDED_EDIT: Replace fixed evaluation-time flip weighting with the best verified adaptive rule, shifting weight toward the higher-margin view only for decisive original/flip disagreements.

EVIDENCE: Reference Design 2 achieved the same 9,331 correct predictions as the current design while improving cross-entropy to 0.1962260711669922 and validation_score to 9331.41798119273.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.184 * (
                0.4914990234375 * logits + 0.5085009765625 * flipped_logits
            )
        return logits
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))

            original_top2 = logits.topk(2, dim=1).values
            flipped_top2 = flipped_logits.topk(2, dim=1).values
            original_margin = original_top2[:, 0] - original_top2[:, 1]
            flipped_margin = flipped_top2[:, 0] - flipped_top2[:, 1]
            margin_difference = flipped_margin - original_margin

            disagreement = logits.argmax(dim=1) != flipped_logits.argmax(dim=1)
            decisive = disagreement & (margin_difference.abs() > 0.5)
            adjustment = (
                0.025
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