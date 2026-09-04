MECHANISM: Quadratically refined cautious Muon decay

HYPOTHESIS: Reducing Muon weight decay from 0.12 to 0.11 will better approach the diminishing-return optimum implied by the 0.20, 0.14, and 0.12 results, lowering val_bpb below 0.984109.

INTENDED_EDIT: Lower only the cautious Muon weight-decay coefficient from 0.12 to 0.11, preserving the best 176K-token batch regime and all other settings.

EVIDENCE: Lowering weight decay from 0.20 to 0.14 improved val_bpb from 0.986967 to 0.984418, and lowering it again to 0.12 improved val_bpb to 0.984109; the smaller second gain indicates an optimum near 0.11 and motivates a fine-grained interpolation.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.12     # refine update-count-normalized cautious decay
=======
WEIGHT_DECAY = 0.11     # refine update-count-normalized cautious decay
>>>>>>> REPLACE