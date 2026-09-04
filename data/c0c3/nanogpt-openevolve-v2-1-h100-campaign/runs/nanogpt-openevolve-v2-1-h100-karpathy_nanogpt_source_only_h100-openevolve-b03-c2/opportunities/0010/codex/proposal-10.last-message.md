MECHANISM: Late-weighted equal-area cooldown

HYPOTHESIS: Redistributing cooldown learning rate from its early half to its late half while preserving endpoints and total exposure will beat the linear schedule’s 0.98713 val_bpb.

INTENDED_EDIT: Replace linear warmdown with a monotonic equal-area curve that decays faster initially and retains larger learning rates during late refinement.

EVIDENCE: The equal-area cosine schedule shifted learning rate earlier and regressed from 0.98713 to 0.990027 at essentially identical throughput, motivating the opposite temporal redistribution as an isolated schedule test.

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        late_weighted_cooldown = cooldown + cooldown * (1 - cooldown) * (1 - 2 * cooldown)
        return late_weighted_cooldown + (1 - late_weighted_cooldown) * FINAL_LR_FRAC
>>>>>>> REPLACE