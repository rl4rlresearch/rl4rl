MECHANISM: Token-horizon-normalized Muon momentum ramp

HYPOTHESIS: Extending the Muon momentum ramp to preserve its 256K-batch token horizon will reduce noisy early updates in the 176K regime and lower val_bpb below 0.984068.

INTENDED_EDIT: Replace the fixed 300-step momentum ramp with an equivalent token-based ramp, reaching 0.95 momentum after 300 × 256K tokens.

EVIDENCE: Normalizing step-dependent weight decay for the 176K batch improved val_bpb from 0.986967 to 0.984418. Unlike the unsuccessful permanent beta2 increase, this change normalizes only the early first-moment transition and restores the verified baseline afterward.

<<<<<<< SEARCH
def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95
=======
def get_muon_momentum(step):
    # Preserve the 300-step momentum-ramp token horizon of the 256K batch regime.
    frac = min(step * TOTAL_BATCH_SIZE / (300 * 16 * 2**14), 1)
    return (1 - frac) * 0.85 + frac * 0.95
>>>>>>> REPLACE