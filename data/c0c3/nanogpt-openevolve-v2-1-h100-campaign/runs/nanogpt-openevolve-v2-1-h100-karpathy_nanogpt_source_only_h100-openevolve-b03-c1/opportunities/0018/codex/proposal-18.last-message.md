MECHANISM: Cosine terminal learning-rate annealing

HYPOTHESIS: Replacing the linear half-run warmdown with a cosine warmdown of equal duration, endpoints, and mean learning rate will reduce val_bpb below 0.985757 by preserving larger updates early in annealing while damping updates more strongly near completion.

INTENDED_EDIT: Preserve the winning 112-sequence batch and all optimizer settings, changing only the warmdown curve from linear to cosine.

EVIDENCE: Batch sizes 108, 112, 113, and 116 produced val_bpb values within 0.000290, with 112 remaining best; this motivates holding batch size fixed and isolating learning-rate schedule geometry.

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine_decay = 0.5 - 0.5 * math.cos(math.pi * cooldown)
        return cosine_decay + (1 - cosine_decay) * FINAL_LR_FRAC
>>>>>>> REPLACE