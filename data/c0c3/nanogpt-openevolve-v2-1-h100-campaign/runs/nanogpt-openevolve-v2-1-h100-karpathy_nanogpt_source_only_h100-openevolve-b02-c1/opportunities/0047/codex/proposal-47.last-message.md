MECHANISM: Integral-matched asymmetric 4.1-power cautious decay

HYPOTHESIS: A 4.1-power Muon decay schedule will lower val_bpb below 0.982576 by shifting slightly toward stronger front-loading while remaining much closer to the quartic optimum than the regressed quintic schedule.

INTENDED_EDIT: Increase the decay exponent from 4.0 to 4.1 and raise peak weight decay from 0.290262 to 0.295940, preserving LR-weighted cumulative decay exposure.

EVIDENCE: Quartic decay achieved the best val_bpb of 0.982576; both 3.9-power decay at 0.982956 and quintic decay at 0.983299 regressed, so a small asymmetric step above 4.0 is the most direct remaining local probe.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.290262 # exposure-matched peak for quartic cautious decay
=======
WEIGHT_DECAY = 0.295940 # exposure-matched peak for 4.1-power cautious decay
>>>>>>> REPLACE

<<<<<<< SEARCH
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 4
=======
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 4.1
>>>>>>> REPLACE