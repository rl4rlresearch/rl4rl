MECHANISM: Back-loaded cosine learning-rate annealing

HYPOTHESIS: Replacing the winning 50%-window linear warmdown with cosine warmdown will reduce val_bpb below 0.987174 by retaining more learning rate early in the decay while suppressing late-stage optimization noise, without affecting throughput.

INTENDED_EDIT: Keep the architecture, batch size, peak learning rates, and warmdown timing unchanged, but reshape the linear warmdown into a cosine curve.

EVIDENCE: Delaying warmdown to the final 25% worsened val_bpb to 0.991963, while starting linear decay after 25% nearly tied the best result at 0.987184; this suggests late learning rates are harmful while moderate reductions in cumulative learning rate are safe.

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine = 0.5 - 0.5 * math.cos(math.pi * cooldown)
        return cosine * 1.0 + (1 - cosine) * FINAL_LR_FRAC
>>>>>>> REPLACE