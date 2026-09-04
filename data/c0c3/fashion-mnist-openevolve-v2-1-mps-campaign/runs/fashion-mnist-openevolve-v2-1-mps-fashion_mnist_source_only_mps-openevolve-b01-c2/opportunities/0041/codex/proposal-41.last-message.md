MECHANISM: Flip-symmetric confidence-adaptive logit fusion

HYPOTHESIS: Samplewise confidence weighting that preserves flip symmetry will exceed 9,330 correct predictions by favoring the more reliable view for each image while retaining the verified 1.15 calibration scale.

INTENDED_EDIT: Replace fixed equal-weight evaluation ensembling with normalized maximum-softmax-confidence weights; training remains unchanged.

EVIDENCE: Fixed 0.60/0.40 canonical-view weighting reduced correct predictions from 9,330 to 9,319, motivating an adaptive fusion that changes decision boundaries without imposing the harmful global orientation bias.

<<<<<<< SEARCH
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.15 * (0.5 * (logits + flipped_logits))
        return logits
=======
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            original_confidence = F.softmax(logits, dim=-1).amax(
                dim=-1,
                keepdim=True,
            )
            flipped_confidence = F.softmax(flipped_logits, dim=-1).amax(
                dim=-1,
                keepdim=True,
            )
            confidence_sum = original_confidence + flipped_confidence
            logits = 1.15 * (
                original_confidence * logits
                + flipped_confidence * flipped_logits
            ) / confidence_sum
        return logits
>>>>>>> REPLACE