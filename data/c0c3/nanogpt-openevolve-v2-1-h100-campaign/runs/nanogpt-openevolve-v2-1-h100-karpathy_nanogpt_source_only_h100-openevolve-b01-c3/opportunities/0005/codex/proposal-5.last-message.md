MECHANISM: Cosine terminal annealing with full Muon polar refinement

HYPOTHESIS: Restoring five-step Muon orthogonalization and replacing linear warmdown with cosine warmdown will preserve the proven 497M-token throughput while reducing val_bpb below 0.995558.

INTENDED_EDIT: Restore the reference-quality five Muon polar iterations and use cosine-shaped LR decay over the existing final 50% training window.

EVIDENCE: Five Muon iterations achieved 0.995558 versus 0.998073 for three iterations at identical 497.0M tokens and 948 steps; batch and attention changes also failed, motivating a schedule-only experiment on the proven design.

<<<<<<< SEARCH
                momentum=0.95, ns_steps=3, beta2=0.95, weight_decay=weight_decay,
=======
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
>>>>>>> REPLACE

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine_decay = 0.5 - 0.5 * math.cos(math.pi * cooldown)
        return cosine_decay * 1.0 + (1 - cosine_decay) * FINAL_LR_FRAC
>>>>>>> REPLACE