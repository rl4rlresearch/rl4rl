MECHANISM: Integral-matched quartic front-loaded cautious decay

HYPOTHESIS: Further concentrating Muon weight decay early with a quartic schedule while preserving LR-weighted cumulative exposure will reduce val_bpb below 0.983066.

INTENDED_EDIT: Replace cubic cautious decay with quartic decay and raise its peak coefficient from 0.233924 to 0.290262 to preserve cumulative LR-weighted decay exposure.

EVIDENCE: Exposure-matched quadratic decay improved val_bpb from 0.984068 to 0.983444, and cubic decay further improved it to 0.983066, motivating another controlled step toward stronger early concentration.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.233924 # exposure-matched peak for cubic cautious decay
=======
WEIGHT_DECAY = 0.290262 # exposure-matched peak for quartic cautious decay
>>>>>>> REPLACE

<<<<<<< SEARCH
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 3
=======
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 4
>>>>>>> REPLACE