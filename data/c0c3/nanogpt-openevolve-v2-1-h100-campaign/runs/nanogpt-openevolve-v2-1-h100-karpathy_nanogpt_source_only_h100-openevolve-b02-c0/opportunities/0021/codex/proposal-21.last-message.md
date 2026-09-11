MECHANISM: Refined front-loaded Muon weight-decay annealing

HYPOTHESIS: A 1.75-power Muon weight-decay schedule will reduce val_bpb below 0.986327 by retaining slightly more beneficial mid-training regularization than the best quadratic schedule without reverting to the inferior linear schedule.

INTENDED_EDIT: Change Muon weight-decay annealing from power 2.0 to power 1.75, leaving its initial strength and all other settings unchanged.

EVIDENCE: Quadratic decay achieved the best val_bpb of 0.986327 versus 0.986664 for linear decay, while powers 2.5 and 3.0 failed to improve it; testing 1.75 brackets the unexplored side of the current optimum.

<<<<<<< SEARCH
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 2
=======
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 1.75
>>>>>>> REPLACE