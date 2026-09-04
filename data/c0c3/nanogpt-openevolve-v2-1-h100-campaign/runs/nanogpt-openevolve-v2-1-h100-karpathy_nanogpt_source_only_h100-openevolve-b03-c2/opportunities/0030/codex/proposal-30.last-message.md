MECHANISM: Quadratically refined AdamW cooldown

HYPOTHESIS: A 36% AdamW cooldown paired with the verified 59% Muon cooldown will beat val_bpb 0.985708.

INTENDED_EDIT: Shorten AdamW’s linear cooldown from 40% to 36% while retaining Muon’s 59% cooldown and all other settings.

EVIDENCE: With Muon fixed at 59%, AdamW cooldowns of 30%, 35%, and 40% produced val_bpb values of 0.986112, 0.985708, and 0.985875; quadratic interpolation places the estimated minimum near 36%.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.40   # test a shorter AdamW cooldown to preserve late refinement updates
=======
WARMDOWN_RATIO = 0.36   # quadratically refined AdamW cooldown near the observed optimum
>>>>>>> REPLACE