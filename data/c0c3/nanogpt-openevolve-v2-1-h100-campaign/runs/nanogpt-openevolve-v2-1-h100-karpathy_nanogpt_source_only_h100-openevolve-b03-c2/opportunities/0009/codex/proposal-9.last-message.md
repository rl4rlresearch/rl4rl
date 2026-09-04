MECHANISM: Endpoint-weighted cosine learning-rate cooldown

HYPOTHESIS: On Reference Design 3’s efficient geometry and 300-step momentum ramp, replacing linear warmdown with an equal-area cosine warmdown will beat val_bpb 0.98713 by preserving larger updates early in cooldown and providing gentler refinement near the end without changing throughput or total learning-rate exposure.

INTENDED_EDIT: Restore the best verified 300-step Muon momentum ramp and change only the warmdown curve from linear to cosine while retaining its duration and endpoints.

EVIDENCE: Reference Design 3 achieved the best val_bpb, 0.98713, at 489.7M tokens; extending its momentum ramp to 600 steps regressed to 0.999525, motivating restoration of the verified ramp and an isolated schedule-shape test that leaves batch geometry and model capacity unchanged.

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine_cooldown = 0.5 - 0.5 * math.cos(math.pi * cooldown)
        return cosine_cooldown + (1 - cosine_cooldown) * FINAL_LR_FRAC
>>>>>>> REPLACE

<<<<<<< SEARCH
def get_muon_momentum(step):
    frac = min(step / 600, 1)
    return (1 - frac) * 0.85 + frac * 0.95
=======
def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95
>>>>>>> REPLACE