MECHANISM: Cosine-shaped AdamW refinement tail

HYPOTHESIS: A cosine 35% AdamW cooldown paired with the verified linear 59% Muon cooldown will reduce val_bpb below 0.985708 by preserving stronger early refinement while tapering updates more gently near completion.

INTENDED_EDIT: Restore AdamW’s best verified 35% cooldown and change only its tail from linear to cosine; retain Muon’s linear 59% schedule.

EVIDENCE: Linear AdamW cooldowns of 30%, 33%, 34.5%, and 36% all underperformed the 35% result of 0.985708, so schedule duration is locally well explored; a cosine tail tests schedule shape while preserving the best cooldown onset and the same average tail multiplier.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.345  # locally interpolated AdamW cooldown near the observed optimum
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
=======
WARMDOWN_RATIO = 0.35   # best verified AdamW cooldown; use a cosine refinement tail
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
>>>>>>> REPLACE

<<<<<<< SEARCH
def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95
=======
def get_adamw_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown_progress = (progress - (1.0 - WARMDOWN_RATIO)) / WARMDOWN_RATIO
        cosine = 0.5 * (1.0 + math.cos(math.pi * cooldown_progress))
        return cosine + (1.0 - cosine) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95
>>>>>>> REPLACE

<<<<<<< SEARCH
    lrm = get_lr_multiplier(progress, WARMDOWN_RATIO)
    muon_lrm = get_lr_multiplier(progress, MUON_WARMDOWN_RATIO)
=======
    lrm = get_adamw_lr_multiplier(progress)
    muon_lrm = get_lr_multiplier(progress, MUON_WARMDOWN_RATIO)
>>>>>>> REPLACE