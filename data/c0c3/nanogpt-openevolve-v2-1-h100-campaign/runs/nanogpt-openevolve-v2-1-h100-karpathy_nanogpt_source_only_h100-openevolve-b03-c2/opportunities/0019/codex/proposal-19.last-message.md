MECHANISM: Group-decoupled linear cooldown

HYPOTHESIS: A 60% Muon cooldown paired with the verified 55% AdamW cooldown will beat val_bpb 0.986663 by reducing late matrix-update magnitude without weakening late embedding refinement.

INTENDED_EDIT: Restore the best 55% cooldown for AdamW groups while beginning Muon’s linear cooldown five percentage points earlier.

EVIDENCE: The Muon-only 5% terminal floor regressed from 0.986663 to 0.988413, showing that additional late Muon update magnitude is harmful; the globally applied 60% cooldown also reached a competitive 0.987466, motivating an isolated longer cooldown for Muon.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.56   # linear refinement over the final 56% of training
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
=======
WARMDOWN_RATIO = 0.55   # verified AdamW refinement over the final 55% of training
MUON_WARMDOWN_RATIO = 0.60 # earlier Muon decay to reduce late matrix updates
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
>>>>>>> REPLACE

<<<<<<< SEARCH
def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
def get_lr_multiplier(progress, warmdown_ratio):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - warmdown_ratio:
        return 1.0
    else:
        cooldown = (1.0 - progress) / warmdown_ratio
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
>>>>>>> REPLACE

<<<<<<< SEARCH
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
=======
    lrm = get_lr_multiplier(progress, WARMDOWN_RATIO)
    muon_lrm = get_lr_multiplier(progress, MUON_WARMDOWN_RATIO)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * (muon_lrm if group['kind'] == 'muon' else lrm)
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
>>>>>>> REPLACE