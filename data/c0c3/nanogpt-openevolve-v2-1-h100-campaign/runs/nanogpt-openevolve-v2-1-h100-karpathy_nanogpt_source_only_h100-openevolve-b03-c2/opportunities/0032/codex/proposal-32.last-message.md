MECHANISM: Locally interpolated AdamW cooldown

HYPOTHESIS: A 34.5% AdamW cooldown paired with the verified 59% Muon cooldown will beat the best observed val_bpb of 0.985708.

INTENDED_EDIT: Shorten AdamW’s linear cooldown from 55% to 34.5%, retaining Muon’s 59% cooldown and all other settings.

EVIDENCE: With Muon fixed at 59%, AdamW cooldowns of 33%, 35%, and 36% produced val_bpb values of 0.986113, 0.985708, and 0.986141; quadratic interpolation of these nearest observations places the estimated minimum near 34.5%.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.55   # verified AdamW refinement over the final 55% of training
MUON_WARMDOWN_RATIO = 0.59 # locally refined Muon decay near the observed optimum
=======
WARMDOWN_RATIO = 0.345  # locally interpolated AdamW cooldown near the observed optimum
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
>>>>>>> REPLACE