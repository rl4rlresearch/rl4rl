MECHANISM: Quadratically refined AdamW-only cooldown

HYPOTHESIS: Pairing the verified 59% Muon cooldown with a 48% AdamW cooldown will beat val_bpb 0.986065.

INTENDED_EDIT: Shorten AdamW’s cooldown from 55% to 48% while preserving Muon’s 59% cooldown and all other settings.

EVIDENCE: With Muon fixed at 59%, AdamW cooldowns of 45%, 50%, and 55% produced val_bpb 0.986100, 0.986065, and 0.986424; quadratic interpolation places the estimated minimum near 48%.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.55   # verified AdamW refinement over the final 55% of training
=======
WARMDOWN_RATIO = 0.48   # quadratically refined AdamW cooldown near the observed optimum
>>>>>>> REPLACE