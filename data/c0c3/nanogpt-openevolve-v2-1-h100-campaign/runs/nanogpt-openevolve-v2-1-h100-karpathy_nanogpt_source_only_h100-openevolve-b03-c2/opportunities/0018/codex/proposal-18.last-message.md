MECHANISM: Muon-only terminal learning-rate floor

HYPOTHESIS: Retaining 5% learning rate only for Muon matrix updates will beat val_bpb 0.986663 by enabling late matrix refinement without the aggressive terminal Adam embedding updates introduced by the unsuccessful all-group floor.

INTENDED_EDIT: Preserve the verified 55% linear cooldown for AdamW groups while giving only Muon groups a 5% terminal learning-rate floor.

EVIDENCE: The all-group 5% floor regressed to 0.988551; because the embedding Adam LR is 0.6 versus 0.04 for Muon matrices, that experiment left a much larger 0.03 terminal embedding LR, motivating a group-isolated test.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.55   # linear refinement over the final 55% of training
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
=======
WARMDOWN_RATIO = 0.55   # linear refinement over the final 55% of training
FINAL_LR_FRAC = 0.0     # AdamW groups decay fully to zero
MUON_FINAL_LR_FRAC = 0.05 # retain a small terminal LR only for matrix updates
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
=======
    for group in optimizer.param_groups:
        group_lrm = lrm
        if group['kind'] == 'muon':
            group_lrm = MUON_FINAL_LR_FRAC + (1 - MUON_FINAL_LR_FRAC) * lrm
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
        group["lr"] = group["initial_lr"] * group_lrm
>>>>>>> REPLACE