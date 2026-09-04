MECHANISM: Front-loaded quadratic Muon weight-decay annealing

HYPOTHESIS: Preserving the successful initial Muon weight decay of 0.1 while reducing its mid-to-late cumulative shrinkage will lower val_bpb below 0.986664.

INTENDED_EDIT: Change Muon weight decay from linear to quadratic decay over the training window, leaving all other settings unchanged.

EVIDENCE: Initial weight decay 0.1 outperformed both 0.2 and 0.05; this isolates decay timing while retaining the best observed early regularization strength.

<<<<<<< SEARCH
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)
=======
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress) ** 2
>>>>>>> REPLACE