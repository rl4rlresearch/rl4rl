MECHANISM: Integral-matched constant cautious Muon decay

HYPOTHESIS: Redistributing the verified decay exposure uniformly across training will preserve useful late-stage regularization and lower val_bpb below 0.984068.

INTENDED_EDIT: Replace the linearly vanishing 0.126 Muon weight decay with a constant 0.077 coefficient, exactly matching its LR-weighted schedule integral under the existing half-window warmdown.

EVIDENCE: Weight decay materially affected val_bpb, with 0.126 reaching 0.984068; neighboring coefficient refinements were noisy, motivating an exposure-matched test of decay timing instead of another amplitude interpolation.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.126    # interpolate the bracketed cautious-decay optimum
=======
WEIGHT_DECAY = 0.077    # integral-matched constant cautious decay
>>>>>>> REPLACE

<<<<<<< SEARCH
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)
=======
def get_weight_decay(progress):
    return WEIGHT_DECAY
>>>>>>> REPLACE