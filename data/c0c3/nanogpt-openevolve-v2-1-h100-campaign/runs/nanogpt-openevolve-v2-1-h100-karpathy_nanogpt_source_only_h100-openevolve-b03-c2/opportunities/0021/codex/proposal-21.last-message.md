MECHANISM: Locally refined Muon-only cooldown

HYPOTHESIS: A 59% Muon cooldown paired with the verified 55% AdamW cooldown will beat val_bpb 0.986574, because the observed 55%, 60%, and 65% Muon cooldown results imply a local optimum slightly below 60%.

INTENDED_EDIT: Begin Muon’s linear cooldown at 41% training progress instead of 40%, leaving architecture, batching, AdamW scheduling, and optimizer settings unchanged.

EVIDENCE: Muon cooldowns of 55%, 60%, and 65% produced val_bpb values of 0.986663, 0.986574, and 0.986730 respectively; quadratic interpolation of these flanking results places the estimated minimum near 59%.

<<<<<<< SEARCH
MUON_WARMDOWN_RATIO = 0.60 # earlier Muon decay to reduce late matrix updates
=======
MUON_WARMDOWN_RATIO = 0.59 # locally refined Muon decay near the observed optimum
>>>>>>> REPLACE