MECHANISM: Mean-preserving front-loaded annealing

HYPOTHESIS: Redistributing learning rate from the first half of the 78% warmdown to its second half, while preserving its duration, endpoints, and mean, will reduce val_bpb below 0.984312.

INTENDED_EDIT: Replace the linear warmdown with a monotonic sinusoidally tilted linear curve that is 0.05 lower early in annealing and 0.05 higher late in annealing.

EVIDENCE: Equal-mean cosine annealing regressed val_bpb from 0.985757 to 0.988710 when it retained larger early-annealing updates, while progressively earlier linear decay improved val_bpb through the 78% schedule; this directly motivates testing the opposite redistribution without changing total learning-rate exposure.

<<<<<<< SEARCH
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        tilted_cooldown = cooldown + 0.05 * math.sin(2 * math.pi * cooldown)
        return tilted_cooldown * (1.0 - FINAL_LR_FRAC) + FINAL_LR_FRAC
>>>>>>> REPLACE