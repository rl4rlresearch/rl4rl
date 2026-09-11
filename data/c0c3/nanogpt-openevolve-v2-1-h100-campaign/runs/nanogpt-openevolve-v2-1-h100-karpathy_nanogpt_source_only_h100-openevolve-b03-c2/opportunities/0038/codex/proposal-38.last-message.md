MECHANISM: X0-path-only Muon-aligned cooldown

HYPOTHESIS: Cooling only `x0_lambdas` over 59% while restoring the verified 35% cooldown for all other AdamW parameters will reduce `val_bpb` below 0.985708.

INTENDED_EDIT: Tag the `x0_lambdas` optimizer group, restore the best shared AdamW cooldown, and apply Muon’s 59% cooldown only to that tagged group.

EVIDENCE: Cooling both scalar groups over 59% reached 0.986025, whereas cooling only `resid_lambdas` reached 0.986415; the relative advantage of the combined split suggests the high-LR `x0_lambdas` schedule may be beneficial and should be isolated against the 0.985708 baseline.

<<<<<<< SEARCH
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
=======
            dict(kind='adamw', schedule='x0', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.40   # test a shorter AdamW cooldown to preserve late refinement updates
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
=======
WARMDOWN_RATIO = 0.35   # best verified cooldown for standard AdamW parameters
X0_WARMDOWN_RATIO = 0.59 # test Muon-aligned cooldown only for x0 residual mixing
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
    x0_lrm = get_lr_multiplier(progress, X0_WARMDOWN_RATIO)
    muon_lrm = get_lr_multiplier(progress, MUON_WARMDOWN_RATIO)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group_lrm = muon_lrm if group['kind'] == 'muon' else (
            x0_lrm if group.get('schedule') == 'x0' else lrm
        )
        group["lr"] = group["initial_lr"] * group_lrm
>>>>>>> REPLACE