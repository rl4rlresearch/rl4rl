MECHANISM: Further-shortened AdamW refinement tail

HYPOTHESIS: Pairing the verified 59% Muon cooldown with a 40% AdamW cooldown will beat val_bpb 0.986065 by preserving productive late embedding and unembedding updates.

INTENDED_EDIT: Shorten AdamW’s linear cooldown from 55% to 40% and restore Muon’s best verified 59% cooldown.

EVIDENCE: With Muon at 59%, shortening AdamW cooldown from 55% to 50% improved val_bpb from 0.986424 to 0.986065, while 45% remained nearly tied at 0.986100; testing 40% determines whether the apparent late-update plateau extends further.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.55   # verified AdamW refinement over the final 55% of training
MUON_WARMDOWN_RATIO = 0.593 # quadratically refined Muon cooldown near the observed optimum
=======
WARMDOWN_RATIO = 0.40   # test a shorter AdamW cooldown to preserve late refinement updates
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
>>>>>>> REPLACE