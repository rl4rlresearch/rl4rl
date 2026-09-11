MECHANISM: Parameter-group-specific AdamW cooldown

HYPOTHESIS: Cooling only the lm_head over 50% of training while retaining the verified 35% cooldown for other AdamW parameters and 59% for Muon will achieve val_bpb below 0.985708.

INTENDED_EDIT: Give the unembedding parameter group an independent 50% cooldown and select AdamW schedules per optimizer group.

EVIDENCE: The shared AdamW sweep found the best result at 35% (0.985708) but also a local best at 50% (0.986065); separating the low-LR lm_head tests whether one shared schedule is masking complementary group optima.

<<<<<<< SEARCH
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
=======
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0,
                 warmdown_ratio=UNEMBEDDING_WARMDOWN_RATIO),
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.35   # further shorten AdamW cooldown to preserve late refinement
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
=======
WARMDOWN_RATIO = 0.35   # verified cooldown for embeddings and scalar parameters
UNEMBEDDING_WARMDOWN_RATIO = 0.50 # independently stabilize the output head
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * (muon_lrm if group['kind'] == 'muon' else lrm)
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
=======
    for group in optimizer.param_groups:
        if group['kind'] == 'muon':
            group_lrm = muon_lrm
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
        else:
            group_lrm = get_lr_multiplier(progress, group.get("warmdown_ratio", WARMDOWN_RATIO))
        group["lr"] = group["initial_lr"] * group_lrm
>>>>>>> REPLACE