MECHANISM: Muon-aligned residual-scalar cooldown

HYPOTHESIS: Keeping embeddings and lm_head on the best verified 35% AdamW cooldown while cooling residual-mixing scalars over 59% will reduce val_bpb below 0.985708.

INTENDED_EDIT: Restore the verified 35% AdamW cooldown, tag both scalar parameter groups, and schedule those scalars with the 59% Muon cooldown.

EVIDENCE: Shared 35% AdamW cooldown achieved the best val_bpb of 0.985708, while independently changing the lm_head or representation-embedding cooldown regressed; the remaining untested AdamW split is the residual scalars, whose direct control of block mixing motivates synchronizing them with Muon matrices.

<<<<<<< SEARCH
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
=======
            dict(kind='adamw', schedule='scalar', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', schedule='scalar', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.50   # shorter AdamW cooldown to preserve late embedding refinement
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
=======
WARMDOWN_RATIO = 0.35   # best verified cooldown for embeddings and lm_head
SCALAR_WARMDOWN_RATIO = 0.59 # align residual-mixing scalars with Muon matrices
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
>>>>>>> REPLACE

<<<<<<< SEARCH
    lrm = get_lr_multiplier(progress, WARMDOWN_RATIO)
    muon_lrm = get_lr_multiplier(progress, MUON_WARMDOWN_RATIO)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * (muon_lrm if group['kind'] == 'muon' else lrm)
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
=======
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
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
>>>>>>> REPLACE