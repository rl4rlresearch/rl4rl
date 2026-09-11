MECHANISM: Independently shortened AdamW refinement tail

HYPOTHESIS: Pairing the best verified 59% Muon cooldown with a 45% AdamW cooldown will beat val_bpb 0.986065 by further preserving productive late embedding and unembedding updates.

INTENDED_EDIT: Shorten only AdamW’s linear cooldown from 50% to 45%, leaving Muon’s 59% cooldown and all other settings unchanged.

EVIDENCE: With Muon near its optimum, shortening AdamW cooldown from 55% to 50% improved val_bpb from 0.986424 to 0.986065; an isolated five-point continuation directly tests whether that trend persists.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.50   # shorter AdamW cooldown to preserve late embedding refinement
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
=======
WARMDOWN_RATIO = 0.45   # further shorten AdamW cooldown to preserve late embedding refinement
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
>>>>>>> REPLACE