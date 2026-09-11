MECHANISM: Cosine terminal learning-rate annealing

HYPOTHESIS: Replacing the linear 50% warmdown with cosine annealing will reduce val_bpb below 0.985730 by retaining higher productive learning rates early in warmdown while providing gentler final convergence.

INTENDED_EDIT: Keep the best Muon weight decay and warmdown duration unchanged, but change the learning-rate warmdown shape from linear to cosine.

EVIDENCE: A 40% linear warmdown nearly matched the best 50% result (0.985770 versus 0.985730), whereas extending warmdown to 60% regressed to 0.986996; this suggests preserving learning rate earlier while still strongly annealing near completion.

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine_decay = 0.5 * (1.0 - math.cos(math.pi * cooldown))
        return cosine_decay + (1 - cosine_decay) * FINAL_LR_FRAC
>>>>>>> REPLACE