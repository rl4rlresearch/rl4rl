MECHANISM: Equal-area late-biased linear warmdown

HYPOTHESIS: Restoring the verified 23/25 peak rates and shifting warmdown learning-rate area from its first half to its second half will process at least 490M tokens and reduce val_bpb below 0.983505.

INTENDED_EDIT: Restore the best coupled peak rates, then add a monotonic zero-area sinusoidal perturbation to linear warmdown, lowering rates early in decay and raising them late without changing its endpoints, duration, or total area.

EVIDENCE: The 23/25 linear baseline achieved 0.983505, whereas equal-area cosine warmdown regressed to 0.986677; cosine raises rates early and lowers them late relative to linear, motivating a similarly sized perturbation in the opposite direction.

<<<<<<< SEARCH
EMBEDDING_LR = 0.6 * 9 / 10      # downward-refined embedding peak LR
UNEMBEDDING_LR = 0.004 * 9 / 10  # downward-refined lm_head peak LR
MATRIX_LR = 0.04 * 9 / 10        # downward-refined Muon matrix peak LR
SCALAR_LR = 0.5 * 9 / 10         # downward-refined scalar peak LR
=======
EMBEDDING_LR = 0.6 * 23 / 25      # best verified embedding peak LR
UNEMBEDDING_LR = 0.004 * 23 / 25  # best verified lm_head peak LR
MATRIX_LR = 0.04 * 23 / 25        # best verified Muon matrix peak LR
SCALAR_LR = 0.5 * 23 / 25         # best verified scalar peak LR
>>>>>>> REPLACE

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        shaped_cooldown = cooldown + 0.1 * math.sin(2 * math.pi * cooldown)
        return shaped_cooldown + (1 - shaped_cooldown) * FINAL_LR_FRAC
>>>>>>> REPLACE