MECHANISM: Moderately broadened disagreement-gated margin-adaptive flip fusion

HYPOTHESIS: Lowering the verified gate threshold from 0.5 to 0.4 will exceed 9,331 correct predictions, or retain 9,331 while reducing cross-entropy below 0.1962260711669922.

INTENDED_EDIT: Restore the verified 0.025 adaptive fusion rule while extending its small higher-margin-view adjustment to original/flip disagreements with margin disparities between 0.4 and 0.5 logits.

EVIDENCE: The verified 0.025 rule at threshold 0.5 retained 9,331 correct and improved cross-entropy to 0.1962260711669922; the reported 0.0275 result further reduced it to 0.1962239959716797 without changing correctness, supporting cautious expansion of the same view-reliability signal.

<<<<<<< SEARCH
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.184 * (
                0.4914990234375 * logits + 0.5085009765625 * flipped_logits
            )
        return logits
=======
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))

            original_top2 = logits.topk(2, dim=1).values
            flipped_top2 = flipped_logits.topk(2, dim=1).values
            original_margin = original_top2[:, 0] - original_top2[:, 1]
            flipped_margin = flipped_top2[:, 0] - flipped_top2[:, 1]
            margin_difference = flipped_margin - original_margin

            disagreement = logits.argmax(dim=1) != flipped_logits.argmax(dim=1)
            decisive = disagreement & (margin_difference.abs() > 0.4)
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