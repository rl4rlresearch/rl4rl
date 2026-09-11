MECHANISM: High-disparity disagreement-gated view fusion

HYPOTHESIS: On images where the original and flipped views predict different classes, modestly favoring the view with a substantially larger top-two margin will exceed 9,331 correct predictions while preserving the best verified fusion exactly for all other images.

INTENDED_EDIT: Restore the independently verified global fusion coefficients, then apply a bounded two-percentage-point weight adjustment only for disagreeing views whose margin difference exceeds 0.5 logits.

EVIDENCE: The best fixed fusion reached 9,331 correct twice, whereas probability-space fusion fell to 9,329; this motivates preserving raw-logit fusion and changing only high-confidence disagreements. The prior adaptive-fusion verification timed out, so its accuracy effect remains unmeasured.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.184 * (
                0.49149951171875 * logits + 0.50850048828125 * flipped_logits
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
                0.02
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