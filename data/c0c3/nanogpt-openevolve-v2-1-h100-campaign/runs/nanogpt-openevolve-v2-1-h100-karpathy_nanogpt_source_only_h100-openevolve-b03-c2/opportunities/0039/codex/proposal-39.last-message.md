MECHANISM: AdamW terminal learning-rate floor

HYPOTHESIS: Retaining 5% of the initial AdamW learning rate at the end, while preserving the verified 35% AdamW and 59% Muon cooldowns, will reduce val_bpb below 0.985708 by allowing embeddings, unembedding, and residual scalars to track late Muon updates.

INTENDED_EDIT: Restore all AdamW groups to the best shared 35% schedule, add a 5% terminal floor only to AdamW, and keep Muon’s linear cooldown ending at zero.

EVIDENCE: The 35% linear AdamW cooldown achieved the best val_bpb of 0.985708; independently rescheduling parameter groups and replacing the tail with cosine both regressed, motivating a shared schedule whose only new variable is a small amount of continued terminal refinement.

<<<<<<< SEARCH
            dict(kind='adamw', schedule='scalar', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', schedule='scalar', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
=======
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.35   # best verified cooldown for embeddings and lm_head
SCALAR_WARMDOWN_RATIO = 0.59 # align residual-mixing scalars with Muon matrices
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
=======
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.35   # best verified shared AdamW cooldown
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
ADAM_FINAL_LR_FRAC = 0.05 # retain small terminal AdamW refinement updates
MUON_FINAL_LR_FRAC = 0.0  # fully cool Muon matrices
>>>>>>> REPLACE

<<<<<<< SEARCH
def get_lr_multiplier(progress, warmdown_ratio):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - warmdown_ratio:
        return 1.0
    else:
        cooldown = (1.0 - progress) / warmdown_ratio
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
def get_lr_multiplier(progress, warmdown_ratio, final_lr_frac):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - warmdown_ratio:
        return 1.0
    else:
        cooldown = (1.0 - progress) / warmdown_ratio
        return cooldown + (1 - cooldown) * final_lr_frac
>>>>>>> REPLACE

<<<<<<< SEARCH
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress, WARMDOWN_RATIO)
    scalar_lrm = get_lr_multiplier(progress, SCALAR_WARMDOWN_RATIO)
    muon_lrm = get_lr_multiplier(progress, MUON_WARMDOWN_RATIO)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group_lrm = muon_lrm if group['kind'] == 'muon' else (
            scalar_lrm if group.get('schedule') == 'scalar' else lrm
        )
        group["lr"] = group["initial_lr"] * group_lrm
=======
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress, WARMDOWN_RATIO, ADAM_FINAL_LR_FRAC)
    muon_lrm = get_lr_multiplier(progress, MUON_WARMDOWN_RATIO, MUON_FINAL_LR_FRAC)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group_lrm = muon_lrm if group['kind'] == 'muon' else lrm
        group["lr"] = group["initial_lr"] * group_lrm
>>>>>>> REPLACE