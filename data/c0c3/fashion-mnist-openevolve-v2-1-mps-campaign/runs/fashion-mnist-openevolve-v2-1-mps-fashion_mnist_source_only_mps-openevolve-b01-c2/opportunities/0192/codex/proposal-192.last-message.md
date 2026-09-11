MECHANISM: Boundary-refined margin-adaptive flip fusion

HYPOTHESIS: Raising the bounded adjustment from 0.0296875 to 0.02984375 will retain 9,331 correct predictions while reducing validation cross-entropy below 0.1962222442626953.

INTENDED_EDIT: Replace fixed evaluation-time flip fusion with disagreement-gated fusion using the midpoint between the best verified adjustment and the unverified 0.03 endpoint.

EVIDENCE: Adjustments from 0.025 through 0.0296875 monotonically reduced cross-entropy while preserving 9,331 correct predictions; 0.02984375 conservatively tests whether that trend continues toward 0.03.

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
                0.02984375
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