MECHANISM: Equal-area backloaded cooldown

HYPOTHESIS: Backloading the proven 78.5% cooldown while preserving its endpoints and integrated learning rate will beat 0.984455 val_bpb by shifting optimization from aggressive early cooldown updates toward later refinement.

INTENDED_EDIT: Blend the linear cooldown halfway toward the reflection of cosine decay around linear, leaving warmdown duration and total learning-rate exposure unchanged.

EVIDENCE: At 50% warmdown, equal-area cosine decay shifted learning rate earlier and regressed val_bpb from 0.985318 to 0.988579; testing the opposite skew at the best 78.5% duration directly probes whether later update mass is beneficial.

<<<<<<< SEARCH
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.785  # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
=======
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.785  # fraction of time budget for LR warmdown
COOLDOWN_BACKLOAD = 0.5 # shift LR mass later without changing cooldown integral
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
>>>>>>> REPLACE

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine_cooldown = 0.5 * (1.0 - math.cos(math.pi * cooldown))
        cooldown = cooldown + COOLDOWN_BACKLOAD * (cooldown - cosine_cooldown)
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
>>>>>>> REPLACE