MECHANISM: Reciprocal representation-embedding cooldown

HYPOTHESIS: A 30% cooldown for token and value embeddings, while retaining the verified 35% cooldown for the lm_head/scalars and 59% for Muon, will reduce val_bpb below 0.985708.

INTENDED_EDIT: Restore the best shared AdamW cooldown, then independently shorten only the representation-embedding cooldown.

EVIDENCE: Shared 35% achieved the best val_bpb of 0.985708, shared 30% regressed to 0.986112, and lengthening only the lm_head cooldown regressed to 0.986361; the reciprocal split tests whether embeddings benefit from 30% while the lm_head requires 35%.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.40   # test a shorter AdamW cooldown to preserve late refinement updates
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
=======
WARMDOWN_RATIO = 0.35   # best verified cooldown for lm_head and scalar parameters
REPRESENTATION_WARMDOWN_RATIO = 0.30 # shorter cooldown for token and value embeddings
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
>>>>>>> REPLACE

<<<<<<< SEARCH
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
=======
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0, warmdown_ratio=REPRESENTATION_WARMDOWN_RATIO),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0, warmdown_ratio=REPRESENTATION_WARMDOWN_RATIO),
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * (muon_lrm if group['kind'] == 'muon' else lrm)
        if group['kind'] == 'muon':
=======
    for group in optimizer.param_groups:
        group_lrm = (muon_lrm if group['kind'] == 'muon'
                     else get_lr_multiplier(progress, group.get("warmdown_ratio", WARMDOWN_RATIO)))
        group["lr"] = group["initial_lr"] * group_lrm
        if group['kind'] == 'muon':
>>>>>>> REPLACE