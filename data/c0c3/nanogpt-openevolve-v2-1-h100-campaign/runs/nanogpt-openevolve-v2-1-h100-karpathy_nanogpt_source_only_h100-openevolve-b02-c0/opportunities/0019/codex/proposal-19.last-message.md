MECHANISM: More front-loaded cubic Muon weight-decay annealing

HYPOTHESIS: Preserving the successful initial Muon weight decay of 0.1 while changing quadratic decay to cubic decay will reduce val_bpb below 0.986327 by further suppressing unproductive late-training shrinkage.

INTENDED_EDIT: Change Muon weight decay from quadratic to cubic decay over the fixed training window, leaving all other settings unchanged.

EVIDENCE: Initial decay 0.1 with quadratic annealing improved val_bpb from 0.986664 to 0.986327, while lowering the initial decay to 0.05 worsened it to 0.987027; cubic annealing retains the beneficial early strength while testing whether still less mid-to-late decay improves validation.

<<<<<<< SEARCH
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 2
=======
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 3
>>>>>>> REPLACE