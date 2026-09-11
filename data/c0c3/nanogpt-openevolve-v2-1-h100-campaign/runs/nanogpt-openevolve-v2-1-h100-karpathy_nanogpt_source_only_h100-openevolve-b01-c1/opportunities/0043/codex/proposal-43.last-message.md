MECHANISM: Cosine learning-rate warmdown

HYPOTHESIS: Replacing linear warmdown with cosine warmdown will lower val_bpb below 0.983689 by sustaining larger productive updates through the middle of training while tapering updates more gently near convergence.

INTENDED_EDIT: Keep the best SMMLSSSL architecture and all optimization magnitudes fixed, changing only the warmdown curve from linear to cosine.

EVIDENCE: SMMLSSSL achieved the best val_bpb of 0.983689, while nearby attention reallocations and more expressive MLP/gating changes regressed; preserving that architecture and testing a throughput-neutral optimization change is the most informative next axis.

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine = 0.5 - 0.5 * math.cos(math.pi * cooldown)
        return cosine + (1 - cosine) * FINAL_LR_FRAC
>>>>>>> REPLACE