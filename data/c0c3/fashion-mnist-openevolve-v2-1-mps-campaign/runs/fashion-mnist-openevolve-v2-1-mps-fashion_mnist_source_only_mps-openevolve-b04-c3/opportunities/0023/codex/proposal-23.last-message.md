MECHANISM: Logit-space flip ensemble with 12.5% cosine floor

HYPOTHESIS: Restoring the verified logit-space flip ensemble and 12.5% terminal learning-rate floor will reproduce the strongest available design, exceeding the current 9,227 correct predictions toward the demonstrated 9,240.

INTENDED_EDIT: Replace probability-space flip fusion with arithmetic logit averaging and change zero-terminal cosine decay to a 12.5%-floor schedule.

EVIDENCE: Reference Design 2 achieved the best available result—9,240 correct with 0.208897 cross-entropy—using these settings; the current probability-mixture, zero-floor design achieved only 9,227 correct.

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