MECHANISM: Earlier Muon momentum smoothing

HYPOTHESIS: Shortening the Muon momentum ramp to 206 steps will reach the noise-suppressing 0.95 momentum sooner and lower val_bpb below 0.984068.

INTENDED_EDIT: Reduce only the Muon momentum-ramp duration from 300 to 206 steps.

EVIDENCE: Extending the ramp from 300 to approximately 436 steps regressed val_bpb from 0.984068 to 0.984170; testing the inverse batch-ratio scaling, 300 × 176/256 ≈ 206, probes the more-promising shorter-ramp direction while preserving every other verified setting.

<<<<<<< SEARCH
def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95
=======
def get_muon_momentum(step):
    frac = min(step / 206, 1)
    return (1 - frac) * 0.85 + frac * 0.95
>>>>>>> REPLACE