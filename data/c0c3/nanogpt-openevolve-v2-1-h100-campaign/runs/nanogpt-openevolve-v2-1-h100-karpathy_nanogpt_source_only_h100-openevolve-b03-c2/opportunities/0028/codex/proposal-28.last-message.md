MECHANISM: Further-shortened AdamW refinement tail with independently cooled Muon matrices

HYPOTHESIS: Pairing the verified 59% Muon cooldown with a 35% AdamW cooldown will beat val_bpb 0.985875 by extending the productive late embedding and unembedding updates indicated by the improvement from 45% to 40%.

INTENDED_EDIT: Decouple optimizer-group schedules, set AdamW cooldown to 35%, and retain Muon’s best verified 59% cooldown.

EVIDENCE: With Muon fixed at 59%, shortening AdamW cooldown from 45% to 40% improved val_bpb from 0.986100 to the best observed 0.985875, leaving 40% as the boundary point worth extending.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.55   # linear refinement over the final 55% of training
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
=======
WARMDOWN_RATIO = 0.35   # further shorten AdamW cooldown to preserve late refinement
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
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