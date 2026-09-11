MECHANISM: Earlier Muon-only linear cooldown

HYPOTHESIS: Extending Muon cooldown from 60% to 65% while restoring AdamW’s verified 55% cooldown will beat val_bpb 0.986574 by further suppressing harmful late matrix updates without reducing late embedding refinement.

INTENDED_EDIT: Apply a 55% linear cooldown to AdamW groups and a 65% linear cooldown to Muon groups, retaining the proven architecture and 262K-token update geometry.

EVIDENCE: Muon-only terminal LR increased val_bpb from 0.986663 to 0.988413, while extending only Muon’s cooldown to 60% improved it to 0.986574; testing 65% determines whether earlier matrix decay continues that improvement.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.6    # longer linear refinement over the final 60% of training
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
=======
WARMDOWN_RATIO = 0.55   # verified AdamW refinement over the final 55% of training
MUON_WARMDOWN_RATIO = 0.65 # earlier Muon decay to further reduce late matrix updates
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
=======
    lrm = get_lr_multiplier(progress, WARMDOWN_RATIO)
    muon_lrm = get_lr_multiplier(progress, MUON_WARMDOWN_RATIO)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * (muon_lrm if group['kind'] == 'muon' else lrm)
>>>>>>> REPLACE