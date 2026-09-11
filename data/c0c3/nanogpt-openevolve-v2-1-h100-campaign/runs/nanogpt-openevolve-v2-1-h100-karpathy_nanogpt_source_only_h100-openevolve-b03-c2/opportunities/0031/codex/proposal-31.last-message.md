MECHANISM: Quadratically refined AdamW cooldown

HYPOTHESIS: A 33% AdamW cooldown paired with the verified 59% Muon cooldown will beat the best observed val_bpb of 0.985708.

INTENDED_EDIT: Shorten only AdamW’s linear cooldown from 50% to 33%, retaining Muon’s 59% cooldown and all other settings.

EVIDENCE: With Muon fixed at 59%, AdamW cooldowns of 30%, 35%, and 36% produced val_bpb values of 0.986112, 0.985708, and 0.986141; quadratic interpolation of these latest observations places the estimated minimum near 33%.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.50   # shorter AdamW cooldown to preserve late embedding refinement
=======
WARMDOWN_RATIO = 0.33   # quadratically refined AdamW cooldown near the observed optimum
>>>>>>> REPLACE