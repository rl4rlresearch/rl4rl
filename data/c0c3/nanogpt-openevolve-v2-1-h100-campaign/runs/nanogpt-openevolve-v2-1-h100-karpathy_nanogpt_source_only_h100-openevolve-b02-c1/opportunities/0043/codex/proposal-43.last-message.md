MECHANISM: Integral-matched cubic front-loaded cautious decay

HYPOTHESIS: Further concentrating Muon weight decay early with a cubic schedule while preserving cumulative LR-weighted exposure will reduce val_bpb below 0.983444.

INTENDED_EDIT: Replace quadratic cautious decay with cubic decay and raise its peak coefficient from 0.17884 to 0.233924 to preserve the verified schedule’s LR-weighted integral.

EVIDENCE: Constant exposure-matched decay regressed to 0.986934, whereas front-loaded quadratic decay improved val_bpb from 0.984068 to 0.983444, directly motivating a controlled test of stronger front-loading.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.17884  # exposure-matched peak for quadratic cautious decay
=======
WEIGHT_DECAY = 0.233924 # exposure-matched peak for cubic cautious decay
>>>>>>> REPLACE

<<<<<<< SEARCH
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 2
=======
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 3
>>>>>>> REPLACE