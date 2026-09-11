MECHANISM: Three-point interpolated Muon momentum ramp

HYPOTHESIS: Extending the Muon momentum ramp from 300 to 362 steps will lower val_bpb below 0.984068 by approaching the empirical minimum bracketed by the verified 206-, 300-, and 436-step ramps.

INTENDED_EDIT: Change only the Muon momentum-ramp duration to 362 optimizer steps, preserving the best architecture, batch size, learning rates, and weight decay.

EVIDENCE: The 300-step ramp achieved 0.984068, while 436 steps remained close at 0.984170 and 206 steps regressed substantially to 0.985398; quadratic interpolation of this asymmetric bracket places the estimated minimum near 362 steps.

<<<<<<< SEARCH
def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95
=======
def get_muon_momentum(step):
    frac = min(step / 362, 1)
    return (1 - frac) * 0.85 + frac * 0.95
>>>>>>> REPLACE