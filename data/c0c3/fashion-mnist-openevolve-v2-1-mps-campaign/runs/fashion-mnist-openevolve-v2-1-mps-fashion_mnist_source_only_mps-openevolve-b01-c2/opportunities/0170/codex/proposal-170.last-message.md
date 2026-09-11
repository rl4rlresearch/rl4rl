MECHANISM: Disagreement-gated margin-adaptive view fusion

HYPOTHESIS: Adjusting fusion only when the original and flipped views disagree, slightly favoring the view with the larger top-two margin, will exceed 9,331 correct predictions while leaving all agreeing-view decisions and training unchanged.

INTENDED_EDIT: Retain the best verified global fusion weights and calibration, but add a bounded per-image fusion adjustment based on the difference between each view’s top-two logit margin.

EVIDENCE: Fixed fusion and calibration have saturated at 9,331 correct, while probability-space fusion fell to 9,329; this tests per-example view reliability without normalizing away logit-scale evidence or perturbing training.

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
            disagree = logits.argmax(dim=1) != flipped_logits.argmax(dim=1)
            margin_adjustment = 0.025 * torch.tanh(
                flipped_margin - original_margin
            )
            flipped_weight = 0.5085009765625 + (
                disagree.to(logits.dtype) * margin_adjustment
            )
            flipped_weight = flipped_weight.unsqueeze(1)
            logits = 1.184 * (
                (1.0 - flipped_weight) * logits
                + flipped_weight * flipped_logits
            )
        return logits
>>>>>>> REPLACE