MECHANISM: Integral-matched front-loaded quadratic cautious decay

HYPOTHESIS: Concentrating the verified decay exposure earlier while reducing late-stage regularization will lower val_bpb below 0.984068.

INTENDED_EDIT: Replace linear Muon decay with a quadratic schedule and raise its initial coefficient to 0.17884, preserving the original LR-weighted cumulative decay exposure.

EVIDENCE: Redistributing the same exposure uniformly with constant 0.077 decay worsened val_bpb from 0.984068 to 0.986934, indicating that stronger early decay and vanishing late decay are beneficial; quadratic decay tests that direction without changing total exposure.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.126    # interpolate the bracketed cautious-decay optimum
=======
WEIGHT_DECAY = 0.17884  # exposure-matched peak for quadratic cautious decay
>>>>>>> REPLACE

<<<<<<< SEARCH
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)
=======
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 2
>>>>>>> REPLACE