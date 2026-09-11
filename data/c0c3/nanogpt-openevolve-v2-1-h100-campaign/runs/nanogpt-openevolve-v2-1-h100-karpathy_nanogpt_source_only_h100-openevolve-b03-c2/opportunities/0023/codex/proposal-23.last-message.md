MECHANISM: Quadratically refined Muon-only cooldown

HYPOTHESIS: A 59.3% Muon cooldown paired with the verified 55% AdamW cooldown will beat the best observed val_bpb of 0.986424.

INTENDED_EDIT: Decouple optimizer-group schedules, preserving AdamW’s 55% cooldown while beginning Muon’s linear cooldown at 40.7% training progress.

EVIDENCE: Muon cooldowns of 58%, 59%, and 60% produced val_bpb values of 0.986991, 0.986424, and 0.986574; quadratic interpolation of these closest observations estimates the minimum near 59.3%.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.55   # linear refinement over the final 55% of training
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
=======
WARMDOWN_RATIO = 0.55   # verified AdamW refinement over the final 55% of training
MUON_WARMDOWN_RATIO = 0.593 # quadratically refined Muon cooldown near the observed optimum
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
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
=======
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress, WARMDOWN_RATIO)
    muon_lrm = get_lr_multiplier(progress, MUON_WARMDOWN_RATIO)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * (muon_lrm if group['kind'] == 'muon' else lrm)
        if group['kind'] == 'muon':
>>>>>>> REPLACE