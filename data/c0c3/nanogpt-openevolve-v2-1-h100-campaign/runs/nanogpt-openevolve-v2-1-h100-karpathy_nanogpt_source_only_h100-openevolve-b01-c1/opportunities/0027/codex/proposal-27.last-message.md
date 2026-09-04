MECHANISM: Equal-area late-biased cubic warmdown

HYPOTHESIS: Shifting learning-rate mass from early to late warmdown while preserving duration, endpoints, and average rate will lower val_bpb below 0.985229.

INTENDED_EDIT: Replace linear warmdown with a monotonic cubic schedule that keeps WARMDOWN_RATIO at 0.625 but redistributes rate toward later updates.

EVIDENCE: The equal-area cosine schedule shifted rate toward early warmdown and away from late updates, regressing val_bpb from 0.985229 to 0.987814; testing the opposite redistribution directly probes whether later warmdown updates are more valuable.

<<<<<<< SEARCH
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        shaped_cooldown = cooldown + cooldown * (1 - cooldown) * (1 - 2 * cooldown)
        return shaped_cooldown * 1.0 + (1 - shaped_cooldown) * FINAL_LR_FRAC
>>>>>>> REPLACE