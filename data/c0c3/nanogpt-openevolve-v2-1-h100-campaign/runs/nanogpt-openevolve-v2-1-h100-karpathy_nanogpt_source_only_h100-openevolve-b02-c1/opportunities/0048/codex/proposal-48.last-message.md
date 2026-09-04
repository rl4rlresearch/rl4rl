MECHANISM: Quadratically interpolated 4.05-power cautious decay

HYPOTHESIS: A 4.05-power Muon decay schedule with matched cumulative exposure will lower val_bpb below 0.982570 by targeting the interpolated minimum between the verified 4.0 and 4.1 schedules.

INTENDED_EDIT: Reduce the decay exponent from 4.1 to 4.05 and adjust peak weight decay from 0.295940 to 0.293101 to preserve LR-weighted cumulative decay exposure.

EVIDENCE: The 4.0 schedule reached 0.982576, 4.1 improved narrowly to 0.982570, and 3.9 regressed to 0.982956; quadratic interpolation places the local minimum near exponent 4.05.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.295940 # exposure-matched peak for 4.1-power cautious decay
=======
WEIGHT_DECAY = 0.293101 # exposure-matched peak for 4.05-power cautious decay
>>>>>>> REPLACE

<<<<<<< SEARCH
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 4.1
=======
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 4.05
>>>>>>> REPLACE