MECHANISM: Logit-space flip ensemble with 12.5%-floor cosine decay

HYPOTHESIS: Restoring the exact strongest verified inference and schedule settings will improve the current 9,227 correct predictions toward the demonstrated 9,240.

INTENDED_EDIT: Replace probability-space flip fusion with arithmetic logit averaging and raise the cosine schedule’s terminal learning rate from zero to 12.5% of peak.

EVIDENCE: Reference Design 2 achieved the best verified result—9,240 correct with 0.208897 cross-entropy—while the current probability-fusion, zero-floor design achieved 9,227 correct.

<<<<<<< SEARCH
            logits = torch.logaddexp(
                F.log_softmax(logits, dim=-1),
                F.log_softmax(flipped_logits, dim=-1),
            ) - math.log(2.0)
=======
            logits = 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE

<<<<<<< SEARCH
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.125 + 0.875 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE