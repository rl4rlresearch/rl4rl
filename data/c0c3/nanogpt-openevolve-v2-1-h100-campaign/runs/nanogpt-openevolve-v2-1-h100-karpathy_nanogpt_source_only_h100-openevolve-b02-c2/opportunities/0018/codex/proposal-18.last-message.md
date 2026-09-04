MECHANISM: Faster Muon momentum stabilization

HYPOTHESIS: Restoring the best 50% linear warmdown and shortening Muon’s momentum ramp from 300 to 150 updates will lower val_bpb below 0.985746 by reaching the proven 0.95 momentum earlier in the short run.

INTENDED_EDIT: Restore the best verified LR schedule and halve only the Muon momentum-ramp duration.

EVIDENCE: Extending the momentum ramp from 300 to 600 updates worsened val_bpb to 0.988827, while the 300-update configuration was better; warmdown experiments also identify 50% linear decay as the best verified schedule at 0.985746.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.6    # begin LR convergence earlier after shorter warmdown regressed
=======
WARMDOWN_RATIO = 0.5    # restore the best verified linear warmdown
>>>>>>> REPLACE

<<<<<<< SEARCH
def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95
=======
def get_muon_momentum(step):
    frac = min(step / 150, 1)
    return (1 - frac) * 0.85 + frac * 0.95
>>>>>>> REPLACE