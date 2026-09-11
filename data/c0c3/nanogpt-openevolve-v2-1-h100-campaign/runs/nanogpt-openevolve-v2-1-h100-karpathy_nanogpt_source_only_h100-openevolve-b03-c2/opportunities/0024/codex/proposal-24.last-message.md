MECHANISM: Independently shortened AdamW refinement tail

HYPOTHESIS: Pairing the best verified 59% Muon cooldown with a 50% AdamW cooldown will beat val_bpb 0.986424 by preserving useful late embedding and unembedding updates while still suppressing late matrix updates.

INTENDED_EDIT: Restore Muon’s best verified 59% cooldown and shorten only AdamW’s cooldown from 55% to 50%.

EVIDENCE: With Muon fixed at a 60% cooldown, shortening AdamW’s cooldown from 60% to 55% improved val_bpb from 0.987466 to 0.986574; this motivates testing another five-point AdamW reduction while using the stronger 59% Muon schedule.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.55   # verified AdamW refinement over the final 55% of training
MUON_WARMDOWN_RATIO = 0.65 # earlier Muon decay to further reduce late matrix updates
=======
WARMDOWN_RATIO = 0.50   # shorter AdamW cooldown to preserve late embedding refinement
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
>>>>>>> REPLACE