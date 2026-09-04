MECHANISM: Integral-matched quintic front-loaded cautious decay

HYPOTHESIS: Further concentrating Muon weight decay early with a quintic schedule while preserving LR-weighted cumulative exposure will reduce val_bpb below 0.982576.

INTENDED_EDIT: Replace quartic cautious decay with quintic decay and raise its peak coefficient from 0.290262 to 0.347275.

EVIDENCE: Exposure-matched quadratic, cubic, and quartic schedules successively improved val_bpb from 0.984068 to 0.983444, 0.983066, and 0.982576, supporting one more controlled increase in early decay concentration.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.290262 # exposure-matched peak for quartic cautious decay
=======
WEIGHT_DECAY = 0.347275 # exposure-matched peak for quintic cautious decay
>>>>>>> REPLACE

<<<<<<< SEARCH
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 4
=======
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 5
>>>>>>> REPLACE