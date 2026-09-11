MECHANISM: Interpolated Muon weight-decay annealing

HYPOTHESIS: A 2.5-power Muon weight-decay schedule will reduce val_bpb below 0.986327 by retaining the beneficial initial decay of 0.1 while interpolating between the nearly tied quadratic and cubic schedules.

INTENDED_EDIT: Change Muon weight-decay annealing from quadratic to power 2.5, leaving all other settings unchanged.

EVIDENCE: Quadratic decay achieved the best val_bpb of 0.986327, outperforming linear decay at 0.986664, while cubic decay remained close at 0.986372; testing the midpoint directly refines the only recently successful optimization axis.

<<<<<<< SEARCH
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 2
=======
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 2.5
>>>>>>> REPLACE