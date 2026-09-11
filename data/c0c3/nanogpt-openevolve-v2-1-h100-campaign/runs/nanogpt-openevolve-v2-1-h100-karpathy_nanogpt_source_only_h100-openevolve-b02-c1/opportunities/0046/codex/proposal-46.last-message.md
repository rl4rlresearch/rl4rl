MECHANISM: Integral-matched interpolated 3.9-power cautious decay

HYPOTHESIS: Setting the decay exponent to 3.9 with matched cumulative exposure will lower val_bpb below 0.982576.

INTENDED_EDIT: Replace quartic Muon decay with a 3.9-power schedule and reduce its peak coefficient to 0.284591, preserving LR-weighted cumulative decay exposure.

EVIDENCE: Cubic decay achieved 0.983066, quartic improved to 0.982576, and quintic regressed to 0.983299; quadratic interpolation of this bracket estimates the optimum near exponent 3.9.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.290262 # exposure-matched peak for quartic cautious decay
=======
WEIGHT_DECAY = 0.284591 # exposure-matched peak for 3.9-power cautious decay
>>>>>>> REPLACE

<<<<<<< SEARCH
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 4
=======
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 3.9
>>>>>>> REPLACE