MECHANISM: Residual-gain-only Muon-aligned cooldown

HYPOTHESIS: Cooling only `resid_lambdas` over 59% while retaining the verified 35% cooldown for `x0_lambdas` and other AdamW parameters will reduce `val_bpb` below 0.985708.

INTENDED_EDIT: Tag only the residual-gain parameter group and give it an independent 59% linear cooldown; leave `x0_lambdas`, embeddings, lm_head, and Muon settings unchanged.

EVIDENCE: Cooling both scalar groups over 59% regressed to 0.986025, but the groups differ by 100× in learning rate, use different beta1 values, and control distinct residual paths; isolating the low-LR residual gains tests whether the combined result was degraded by the high-LR `x0_lambdas` schedule.

<<<<<<< SEARCH
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
=======
            dict(kind='adamw', schedule='resid', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.35   # further shorten AdamW cooldown to preserve late refinement
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
=======
WARMDOWN_RATIO = 0.35   # best verified cooldown for embeddings, lm_head, and x0 scalars
RESID_WARMDOWN_RATIO = 0.59 # align only residual gains with Muon matrices
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
>>>>>>> REPLACE

<<<<<<< SEARCH
    lrm = get_lr_multiplier(progress, WARMDOWN_RATIO)
    muon_lrm = get_lr_multiplier(progress, MUON_WARMDOWN_RATIO)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * (muon_lrm if group['kind'] == 'muon' else lrm)
=======
    lrm = get_lr_multiplier(progress, WARMDOWN_RATIO)
    resid_lrm = get_lr_multiplier(progress, RESID_WARMDOWN_RATIO)
    muon_lrm = get_lr_multiplier(progress, MUON_WARMDOWN_RATIO)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group_lrm = muon_lrm if group['kind'] == 'muon' else (
            resid_lrm if group.get('schedule') == 'resid' else lrm
        )
        group["lr"] = group["initial_lr"] * group_lrm
>>>>>>> REPLACE