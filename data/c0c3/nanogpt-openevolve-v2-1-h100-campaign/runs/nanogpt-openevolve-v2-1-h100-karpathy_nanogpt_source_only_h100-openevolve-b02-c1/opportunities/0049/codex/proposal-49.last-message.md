MECHANISM: Fixed-shape cumulative cautious-decay increase

HYPOTHESIS: Increasing the 4.05-power Muon decay exposure by 2% will strengthen beneficial early regularization and lower val_bpb below 0.982406.

INTENDED_EDIT: Raise peak Muon weight decay from 0.293101 to 0.298963 while retaining the verified 4.05-power timing.

EVIDENCE: The 4.05-power schedule outperformed the matched-exposure 4.0 and 4.1 schedules, but all exponent probes held cumulative exposure fixed; this tests the remaining orthogonal decay-amplitude axis without confounding schedule shape.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.293101 # exposure-matched peak for 4.05-power cautious decay
=======
WEIGHT_DECAY = 0.298963 # 2% higher exposure with 4.05-power cautious decay
>>>>>>> REPLACE