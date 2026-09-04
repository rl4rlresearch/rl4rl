MECHANISM: Confidence-adaptive symmetric flip fusion

HYPOTHESIS: Softly favoring the more confident orientation at evaluation will correct at least one borderline disagreement and increase validation_correct above 9,330 without changing training or parameters.

INTENDED_EDIT: Replace equal flip-logit averaging with symmetric confidence-weighted averaging while retaining the verified 1.184 calibration scale.

EVIDENCE: The current equal-weight ensemble verifies 9,330 correct, while temperature scaling cannot change argmax predictions; prior alternative fusion attempts produced no completed metrics, so adaptive view weighting is an untested, training-cost-free way to target correctness.

<<<<<<< SEARCH
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.184 * (0.5 * (logits + flipped_logits))
        return logits
=======
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            original_confidence = F.softmax(logits, dim=-1).amax(dim=-1)
            flipped_confidence = F.softmax(
                flipped_logits,
                dim=-1,
            ).amax(dim=-1)
            original_weight = torch.sigmoid(
                4.0 * (original_confidence - flipped_confidence)
            ).unsqueeze(-1)
            logits = 1.184 * (
                original_weight * logits
                + (1.0 - original_weight) * flipped_logits
            )
        return logits
>>>>>>> REPLACE