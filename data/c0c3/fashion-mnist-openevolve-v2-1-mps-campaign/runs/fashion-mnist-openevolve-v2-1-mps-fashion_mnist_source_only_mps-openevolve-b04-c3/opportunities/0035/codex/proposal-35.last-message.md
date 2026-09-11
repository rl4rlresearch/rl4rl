MECHANISM: Verified logit-space flip ensemble with nonzero-floor cosine decay

HYPOTHESIS: Restoring the exact Reference Design 2 inference and schedule settings will improve validation correctness from 9,227 toward the demonstrated 9,240.

INTENDED_EDIT: Replace probability-space flip fusion with arithmetic logit averaging and set the cosine schedule’s terminal learning rate to 12.5% of peak.

EVIDENCE: Reference Design 2 achieved the strongest verified result—9,240 correct with 0.208897 cross-entropy—while the current probability-fusion, zero-floor implementation achieved 9,227; unsuccessful replication attempts provide no contrary validation result.

<<<<<<< SEARCH
        if not self.training:
            flipped_logits = self._classify(images.flip(-1))
            logits = torch.logaddexp(
                F.log_softmax(logits, dim=-1),
                F.log_softmax(flipped_logits, dim=-1),
            ) - math.log(2.0)
        return logits
=======
        if not self.training:
            flipped_logits = self._classify(images.flip(-1))
            logits = 0.5 * (logits + flipped_logits)
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.125 + 0.875 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE